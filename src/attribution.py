from typing import Dict

import torch
from tqdm import trange
from nnsight import LanguageModel

def integrated_gradients(
    model: LanguageModel,
    input_text: str,
    target_token_id: int,
    baseline_id: int | None = None,
    interpolation_steps: int = 50
) -> Dict:
    """
    Calculate the integradient gradients according to https://arxiv.org/pdf/1703.01365

    Args:
        model: nnsight.LanguageModel
        input_text: a prompt truncated at an expected next token
        target_token_id: an expected next token
        baseline_id: the baseline token id, if none set defaults to pad token
        interpolation_steps: number of steps to take for integral approximation
    """
    if baseline_id is None:
        baseline_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
        
    # Get the baseline embedding
    baseline_embed = model.model.embed_tokens.weight[baseline_id].detach()
    
    # Tokenize the full input
    token_ids = model.tokenizer.encode(input_text, add_special_tokens=False)
    tokens = model.tokenizer.tokenize(input_text)
    
    # Get all token embeddings: shape (seq_len, hidden_dim)
    token_embeds = model.model.embed_tokens.weight[token_ids].detach().to(device="cpu")
    
    # Baseline is repeated for each position: shape (seq_len, hidden_dim)
    baseline_embeds = baseline_embed.unsqueeze(0).expand_as(token_embeds).detach().to(device="cpu")
    
    # Difference between input and baseline
    x_minus_baseline = token_embeds - baseline_embeds # (seq_len, hidden_dim)
    
    # Accumulate gradients across interpolation steps
    accumulated_grads = torch.zeros_like(token_embeds).to(device="cpu")
    print(accumulated_grads.device)
    
    for step in trange(1, interpolation_steps + 1):
        alpha = step / interpolation_steps
        interpolated_embeds = baseline_embeds + alpha * x_minus_baseline
        
        with model.trace(input_text):
            # Move to correct device and add batch dimension INSIDE trace
            interpolated_embeds_traced = interpolated_embeds.unsqueeze(0).to(model.device).requires_grad_(True)
            
            # Override the embedding output
            model.model.embed_tokens.output = interpolated_embeds_traced
            
            # Get logits
            logits = model.output[0]
            target_logit = logits[0, -1, target_token_id]
            
            # Compute gradients
            target_logit.backward()
            grad = interpolated_embeds_traced.grad.save()
            
        #print(f"{grad.device=}")
        
        accumulated_grads += grad.squeeze(0)
        
    # Average the gradients
    avg_grads = accumulated_grads / interpolation_steps
    
    # Integrated gradients = (x - x') * avg_grads
    ig_attributions = x_minus_baseline * avg_grads # (seq_len, hidden_dim)
    
    # sum across hidden dimension to get per-token attributino
    token_attributions = ig_attributions.sum(dim=-1) # (seq_len,)
    
    return {
        'tokens': tokens,
        'token_ids': token_ids,
        "attributions": token_attributions,
        "attributions_full": ig_attributions,
    }
