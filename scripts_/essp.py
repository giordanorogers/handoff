import os
import re
import json
import torch
import random
import argparse
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

from src.utils import MathVerifier, CoTSplitter

SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant. "
    "Output your final answer inside \\boxed{}."
)

TRIGGER_PHRASE = "\n</think>\nTherefore, the final answer is \\boxed{"

class ESSPExperiment:
    """Early stopping success point experiment."""
    
    def __init__(self, *args):
        
        # Initialize vLLM
        print(f"Loading {args.model_name}...")
        
        self.llm = LLM(
            model=args.model_name,
            seed=args.seed,
            enable_prefix_caching=True,
            trust_remote_code=True,
            gpu_memory_utilization=0.9
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        self.params_handoff = SamplingParams(
            n=args.n_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed
        )
        
    def extract_reasoning_body(self, raw_cot):
        """
        Extracts text inside <think>...</think>
        If not tags, returns the whole text (fallback).
        """
        match = re.search(r"<think>(.*?)</think>", raw_cot, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: try to remove the \boxed{} part at the end
        print("WARNING: Couldn't extract text from <think> tags. Falling back to raw_cot")
        if "\\boxed" in raw_cot:
            return raw_cot.split("\\boxed")[0].strip()
        return raw_cot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--seed", type=int, default=9001, help="Random seed")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="HF ID")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--threshold_tau", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--max_tokens", type=float, default=31_000)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    experiment = ESSPExperiment()
    experiment.run_experiment()
