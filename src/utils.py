import re
import json

import torch
from transformers import AutoConfig

def create_layer_device_map(model_name: str, reserved_memory_per_gpu: float = 10.0):
    """
    Create a detailed device map that assigns each layer to a specific GPU.
    
    Args:
        model_name: HuggingFace model name
        reserved_memory_per_gpu: GB to reserve per GPU for activations/gradients
    
    Returns:
        dict: Detailed device_map mapping each module to a device
    """
    # Get available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return "cpu"
    
    print(f"Found {num_gpus} GPUs")
    
    # Get model config to find number of layers
    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    print(f"Model has {num_layers} layers")
    
    # Calculate memory per GPU
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_memory_gb = props.total_memory / (1024**3)
        available_memory_gb = total_memory_gb - reserved_memory_per_gpu
        print(f"GPU {i}: {total_memory_gb:.1f}GB total, {available_memory_gb:.0f}GB available for model")
    
    # Distribute layers evenly across GPUs
    layers_per_gpu = num_layers // num_gpus
    remainder = num_layers % num_gpus
    
    device_map = {}
    
    # Embedding layer on first GPU
    device_map["model.embed_tokens"] = 0
    
    # Distribute transformer layers
    current_layer = 0
    for gpu_id in range(num_gpus):
        # Give extra layers to first GPUs if there's a remainder
        num_layers_this_gpu = layers_per_gpu + (1 if gpu_id < remainder else 0)
        
        for i in range(num_layers_this_gpu):
            device_map[f"model.layers.{current_layer}"] = gpu_id
            current_layer += 1
    
    # Put final norm and LM head on last GPU
    device_map["model.norm"] = num_gpus - 1
    device_map["lm_head"] = num_gpus - 1
    
    # Print distribution
    for gpu_id in range(num_gpus):
        layers_on_gpu = [k for k, v in device_map.items() if v == gpu_id]
        print(f"GPU {gpu_id}: {len(layers_on_gpu)} modules")
    
    return device_map


def load_model_with_layer_device_map(model_name: str, reserved_memory_per_gpu: float = 10.0):
    """
    Load an NNsight LanguageModel with explicit layer-by-layer device mapping.
    
    Args:
        model_name: HuggingFace model name
        reserved_memory_per_gpu: GB to reserve per GPU for activations/gradients
    
    Returns:
        LanguageModel with proper device mapping
    """
    from nnsight import LanguageModel
    
    device_map = create_layer_device_map(model_name, reserved_memory_per_gpu)
    
    if device_map == "cpu":
        print("No GPUs found, loading to CPU")
        model = LanguageModel(model_name, device_map="cpu")
    else:
        print(f"Loading model with custom device map...")
        model = LanguageModel(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    
    return model

def load_dataset(input_file):
    dataset = []
    with open(input_file, 'r') as f:
        for line in f:
            try:
                dataset.append(json.loads(line))
            except:
                pass
    return dataset

class MathVerifier:
    """Answer extractor with normalization."""
    
    @staticmethod
    def extract_answer(text):
        if not text: return None
        # Handle cases where whitespace is inserted like \boxed { content }
        start_indices = [m.start() for m in re.finditer(r'\\boxed\s*\{', text)]
        if not start_indices: return None
        
        for start in start_indices:
            balance = 0
            # Find the first '{' after the \boxed
            content_start = text.find('{', start) + 1
            for i in range(content_start, len(text)):
                char = text[i]
                if char == '{': balance += 1
                elif char == '}':
                    if balance == 0: return text[content_start:i]
                    balance -= 1
        return None

    @staticmethod
    def extract_from_partial(text):
        """
        Extracts content when the prompt ends with \boxed{
        We look for the first closing brace '}' that isn't balanced by an opening '{'
        within the generated text itself.
        """
        if not text: return None
        balance = 0
        for i, char in enumerate(text):
            if char == '{':
                balance += 1
            elif char == '}':
                # If balance is 0, this '}' closes the ghost \boxed{ from the prompt
                if balance == 0:
                    return text[:i]
                balance -= 1
        # Fallback: if no closing brace found, return everything (model stopped early)
        return text

    @staticmethod
    def normalize_answer(text):
        if text is None: return ""
        text = text.strip().replace(" ", "")
        
        # --- FIX 1: Handle JSON-style double escaping ---
        # Turns \\left into \left, \\dfrac into \dfrac
        text = text.replace("\\\\", "\\") 
        
        # --- FIX 2: Standardize LaTeX fractions ---
        # Handle all fraction formats: \dfrac, \tfrac, dfrac (missing backslash)
        text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
        text = text.replace("dfrac", "frac")  # Handle cases where backslash was stripped
        
        # --- FIX 3: Remove sizing commands ---
        text = text.replace(r"\left", "").replace(r"\right", "")
        
        # --- FIX 4: Remove \text{...} wrappers ---
        text = re.sub(r'\\text\{(.*?)\}', r'\1', text)
        
        # --- FIX 5: Normalize plain fractions to LaTeX format ---
        # Convert "7/20" to "frac{7}{20}" for consistent comparison
        # This handles simple cases like "7/20", "12/5", etc.
        text = re.sub(r'(\d+)/(\d+)', r'frac{\1}{\2}', text)
        
        return text

    @staticmethod
    def is_correct(generated_text, ground_truth, is_partial=False):
        if is_partial:
            pred = MathVerifier.extract_from_partial(generated_text)
        else:
            pred = MathVerifier.extract_answer(generated_text)
            
        truth = MathVerifier.extract_answer(ground_truth)
        if truth is None: truth = ground_truth
        
        # Debug print to verify the fix works in your logs
        print()
        print(f"  Norm Pred: '{MathVerifier.normalize_answer(pred)}'")
        print(f"  Norm Truth: '{MathVerifier.normalize_answer(truth)}'")

        return MathVerifier.normalize_answer(pred) == MathVerifier.normalize_answer(truth)

class CoTSplitter:
    """
    Splits reasoning text into logical steps (sentences), 
    but protects LaTeX math environments from being split.
    """
    @staticmethod
    def split(text):
        # 1. Identify chunks: Math vs Text
        # Regex to find LaTeX patterns: $...$, $$...$$, \[...\], \(...\)
        # We perform a split that keeps the delimiters
        pattern = r'(\$\$[\s\S]*?\$\$|\$[\s\S]*?\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\))'
        chunks = re.split(pattern, text)
        
        steps = []
        current_step = ""
        
        # 2. Iterate chunks
        for chunk in chunks:
            # If this chunk is a LaTeX block, treat it as a single indivisible atom
            if re.match(pattern, chunk):
                current_step += chunk
                continue
            
            # If it's text, we can split by sentence delimiters
            # We look for punctuation followed by whitespace or end of string
            # We avoid splitting abbreviations (simplified check)
            sub_parts = re.split(r'([.?!]\s+)', chunk)
            
            for part in sub_parts:
                current_step += part
                # If this part ends with a delimiter, it might be a split point
                if re.match(r'[.?!]\s+', part):
                    # Check if the current step is long enough to be a sentence
                    if len(current_step.strip()) > 5:
                        steps.append(current_step.strip())
                        current_step = ""
        
        if current_step.strip():
            steps.append(current_step.strip())
            
        return steps