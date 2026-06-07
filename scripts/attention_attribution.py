"""scripts/attention_attribution.py

Compute attention-based attribution scores for sentences in reasoning traces.
Uses last layer attention, averaged across heads, summed within sentences.

Memory-efficient: Only computes attention for the last layer.
For long sequences, computes attention on CPU to avoid OOM.

Usage:
    CUDA_VISIBLE_DEVICES=7 python -m scripts.attention_attribution
"""

import os
import sys
import json
import logging
import torch
import random
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 9001
MODEL_NAME = "Qwen/Qwen3-32B"
INPUT_FILE = "data/hsp_step_accuracies.jsonl"
OUTPUT_DIR = "data/attention_attribution_results"
LOG_DIR = "logs/attention_attribution"

TRIGGER_PHRASE = "\n</think>\nTherefore, the final answer is \\boxed{"
SYSTEM_PROMPT = "You are a helpful reasoning assistant. Output your final answer inside \\boxed{}."


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"attention_{timestamp}.log")
    root_logger = logging.getLogger()
    root_logger.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    logging.info(f"Logging to: {log_file}")
    return log_file


class AttentionAttribution:
    def __init__(self):
        logging.info(f"Loading model: {MODEL_NAME}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.model.eval()
        
        # Get dimensions from config
        self.num_heads = self.model.config.num_attention_heads
        self.num_kv_heads = self.model.config.num_key_value_heads
        
        # Get head_dim from actual projection weight shapes
        layer0_attn = self.model.model.layers[0].self_attn
        q_out = layer0_attn.q_proj.out_features
        k_out = layer0_attn.k_proj.out_features
        
        self.head_dim = q_out // self.num_heads
        kv_head_dim = k_out // self.num_kv_heads
        
        assert q_out == self.num_heads * self.head_dim
        assert k_out == self.num_kv_heads * kv_head_dim
        assert self.head_dim == kv_head_dim
        
        # Rotary embedding is on the base model, shared across layers
        self.rotary_emb = self.model.model.rotary_emb
        
        logging.info("Model loaded successfully.")
        logging.info(f"Model device: {self.model.device}")
        logging.info(f"Attention: {self.num_heads} Q heads, {self.num_kv_heads} KV heads, {self.head_dim} head_dim")
        logging.info(f"Projection shapes: q_proj -> {q_out}, k_proj -> {k_out}")
        logging.info(f"Rotary embedding: {type(self.rotary_emb).__name__}")

    def _build_base_prompt(self, problem):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _get_token_ranges_by_construction(self, problem, steps, essp_index):
        base_prompt = self._build_base_prompt(problem)
        base_with_think = base_prompt + "<think>\n"
        base_tokens = self.tokenizer.encode(base_with_think, add_special_tokens=True)
        
        logging.info(f"  Base prompt tokens: {len(base_tokens)}")
        
        sentence_ranges = []
        all_tokens = base_tokens.copy()
        
        for i in range(essp_index + 1):
            current_reasoning = steps[i] if i == 0 else " ".join(steps[:i+1])
            full_text = base_with_think + current_reasoning
            full_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
            sentence_ranges.append((len(all_tokens), len(full_tokens)))
            all_tokens = full_tokens
        
        if "\\boxed" not in steps[essp_index]:
            full_text_with_trigger = base_with_think + " ".join(steps[:essp_index+1]) + TRIGGER_PHRASE
            all_tokens = self.tokenizer.encode(full_text_with_trigger, add_special_tokens=True)
            logging.info(f"  Added trigger phrase, total tokens: {len(all_tokens)}")
        
        return torch.tensor([all_tokens], dtype=torch.long), sentence_ranges

    def _compute_attention_for_last_layer(self, hidden_states, layer):
        """Manually compute attention weights for a single layer."""
        attn = layer.self_attn
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim
        
        # Pre-norm
        normed_hidden = layer.input_layernorm(hidden_states)
        
        # Project Q and K
        query_states = attn.q_proj(normed_hidden)
        key_states = attn.k_proj(normed_hidden)
        
        # Reshape to (bsz, num_heads, seq, head_dim)
        query_states = query_states.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        
        # Rotary embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cos, sin = self.rotary_emb(query_states, position_ids)
        
        rotary_dim = cos.shape[-1]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        
        if rotary_dim < head_dim:
            q_rot, q_pass = query_states[..., :rotary_dim], query_states[..., rotary_dim:]
            k_rot, k_pass = key_states[..., :rotary_dim], key_states[..., rotary_dim:]
            q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
            k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
            query_states = torch.cat([q_rot, q_pass], dim=-1)
            key_states = torch.cat([k_rot, k_pass], dim=-1)
        else:
            query_states = (query_states * cos) + (rotate_half(query_states) * sin)
            key_states = (key_states * cos) + (rotate_half(key_states) * sin)
        
        # Expand KV heads for GQA
        if num_kv_heads != num_heads:
            key_states = key_states.repeat_interleave(num_heads // num_kv_heads, dim=1)
        
        del normed_hidden, cos, sin
        torch.cuda.empty_cache()
        
        # Estimate attention matrix size
        attn_matrix_size_gb = (num_heads * seq_len * seq_len * 4) / (1024**3)
        
        # For long sequences, compute on CPU to avoid OOM
        if attn_matrix_size_gb > 2.0:
            logging.info(f"    Large attention ({attn_matrix_size_gb:.1f} GB), computing on CPU...")
            query_states = query_states.float().cpu()
            key_states = key_states.float().cpu()
            torch.cuda.empty_cache()
            
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
            del query_states, key_states
            
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
            attn_weights = attn_weights + causal_mask
            attn_weights = torch.softmax(attn_weights, dim=-1)
            del causal_mask
            
            return attn_weights
        
        # For shorter sequences, compute on GPU
        scale = head_dim ** -0.5
        attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(-2, -1)) * scale
        del query_states, key_states
        torch.cuda.empty_cache()
        
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
        attn_weights = attn_weights + causal_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        attn_weights_cpu = attn_weights.cpu()
        del attn_weights, causal_mask
        torch.cuda.empty_cache()
        
        return attn_weights_cpu

    def _compute_attention_scores(self, problem, steps, essp_index):
        input_ids, sentence_ranges = self._get_token_ranges_by_construction(problem, steps, essp_index)
        input_ids = input_ids.to(self.model.device)
        seq_len = input_ids.shape[1]
        
        logging.info(f"  Total sequence length: {seq_len} tokens")
        logging.info(f"  Number of sentences: {len(sentence_ranges)}")
        
        for i, (start, end) in enumerate(sentence_ranges):
            if start >= seq_len or end > seq_len or start >= end:
                raise ValueError(f"Invalid token range for sentence {i}: [{start}, {end}), seq_len={seq_len}")
        
        logging.info(f"  Running forward pass to get hidden states...")
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_attentions=False,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True
            )
            hidden_states_for_last_layer = outputs.hidden_states[-2].clone()
            del outputs
            torch.cuda.empty_cache()
        
        logging.info(f"  Computing attention for last layer...")
        
        last_layer = self.model.model.layers[-1]
        try:
            attn_weights = self._compute_attention_for_last_layer(hidden_states_for_last_layer, last_layer)
        finally:
            del hidden_states_for_last_layer
            torch.cuda.empty_cache()
        
        logging.info(f"  Attention shape: {attn_weights.shape}")
        
        # Average attention to last token across all heads
        avg_attention = attn_weights[0, :, -1, :].mean(dim=0).detach().numpy()
        logging.info(f"  Attention sum: {avg_attention.sum():.4f} (should be ~1.0)")
        
        del attn_weights
        
        # Aggregate by sentence
        sentence_scores = [float(avg_attention[s:e].sum()) for s, e in sentence_ranges]
        total = sum(sentence_scores)
        normalized_scores = [s/total for s in sentence_scores] if total > 0 else [0.0]*len(sentence_scores)
        
        return sentence_scores, normalized_scores, sentence_ranges

    def run_attribution(self, data, problem_index):
        problem = data.get("problem")
        steps = data.get("steps")
        essp_index = data.get("first_essp_index")
        problem_id = data.get("id", f"problem_{problem_index}")
        
        logging.info("")
        logging.info("=" * 60)
        logging.info(f"Processing problem {problem_id} (index {problem_index})")
        logging.info("=" * 60)
        
        if problem is None or not steps or essp_index in (None, -1) or essp_index >= len(steps):
            logging.warning(f"  Skipping: invalid data")
            return None
        
        logging.info(f"  ESSP index: {essp_index}, Total steps: {len(steps)}, Sentences to process: {essp_index + 1}")
        
        raw_scores, normalized_scores, token_ranges = self._compute_attention_scores(problem, steps, essp_index)
        
        sentence_data = [{
            "tokens": self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(steps[i], add_special_tokens=False)),
            "attention_raw": raw_scores[i],
            "attention_normalized": normalized_scores[i],
            "token_range": list(token_ranges[i]),
            "orig_idx": i
        } for i in range(essp_index + 1)]
        
        result = {
            "problem_id": problem_id, "problem_index": problem_index, "essp_index": essp_index,
            "num_sentences": essp_index + 1, "sentence_attributions": sentence_data,
            "raw_scores": raw_scores, "normalized_scores": normalized_scores,
        }
        
        logging.info("  Top 5 sentences by attention:")
        for rank, idx in enumerate(np.argsort(normalized_scores)[::-1][:5]):
            logging.info(f"    #{rank+1} [sent {idx}] score={normalized_scores[idx]:.4f}: '{steps[idx][:50]}...'")
        
        return result

    def run_experiment(self):
        logging.info(f"Loading data from {INPUT_FILE}...")
        if not os.path.exists(INPUT_FILE):
            logging.error(f"Input file not found: {INPUT_FILE}")
            sys.exit(1)
        
        dataset = []
        with open(INPUT_FILE, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        dataset.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        
        logging.info(f"Loaded {len(dataset)} problems.")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        results = []
        for i, data in enumerate(dataset):
            try:
                result = self.run_attribution(data, i)
            except Exception as e:
                logging.exception(f"Error processing problem {i}: {e}")
                result = None
            
            if result:
                results.append(result)
                with open(os.path.join(OUTPUT_DIR, f"problem_{i}_attention.json"), "w") as f:
                    json.dump(result, f, indent=2)
                logging.info(f"  Saved: problem_{i}_attention.json")
        
        if results:
            combined_file = os.path.join(OUTPUT_DIR, f"all_attention_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(combined_file, "w") as f:
                json.dump(results, f, indent=2)
            logging.info(f"Combined results saved: {combined_file}")
        
        logging.info(f"COMPLETED: {len(results)}/{len(dataset)} problems processed successfully")
        return results


def main():
    log_file = setup_logging()
    logging.info("=" * 60)
    logging.info("ATTENTION ATTRIBUTION EXPERIMENT")
    logging.info("=" * 60)
    logging.info(f"Model: {MODEL_NAME}, Input: {INPUT_FILE}, Output: {OUTPUT_DIR}, Seed: {SEED}")
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    AttentionAttribution().run_experiment()


if __name__ == "__main__":
    main()