"""
int_grad.py

NOTE: CHECK TODO's!
"""

import json
import numpy as np
from nnsight import LanguageModel

from src.utils import load_dataset
from src.attribution import integrated_gradients

MODEL_NAME = "Qwen/Qwen3-32B"

INPUT_FILE = "data/_.jsonl"
OUTPUT_FILE = "data/_.jsonl"

PRUNE = False
TAU = 0.8
BETA = 0.8

class IGAExperiment:
    "Integrated gradient attribution experiment."
    
    def __init__(self):
        
        print(f"Loading Model: {MODEL_NAME}...")
        self.model = LanguageModel(MODEL_NAME)
        self.tokenizer = self.model.tokenizer
        
    def extract_cot_minus_answer(self, full_cot):
        # Calculate the index of the last character of "\\boxed{"
        index_of_boxed = full_cot.rfind("\\boxed{", )
        index_of_answer = index_of_boxed + +len("\\boxed{")
        return full_cot[:index_of_answer]

    def get_target_token_id(self, full_cot, cot_minus_answer):
        """Get the id of the first token of the final answer."""
        token_ids = self.tokenizer.encode(full_cot, add_special_tokens=False)
        prefix_tokens = self.tokenizer.encode(cot_minus_answer, add_special_tokens=False)
        target_token_id = token_ids[len(prefix_tokens)]
        return target_token_id
    
    def normalize_segment_strengths(self, segment_data):
        total_strength = 0
        for item in segment_data:
            total_strength += item['normalized_strength_1']
        for item in segment_data:
            normalized_strength_1 = item['normalized_strength_1']
            normalized_strength_2 = normalized_strength_1 / total_strength
        return normalized_strength_2
    
    def segment_strengths_and_consistencies(self, int_grad_result: dict):
        tokens = int_grad_result['tokens']
        attributions = int_grad_result['attributions']
        
        # TODO: Likely need to increase this for new samples
        delimiters = {'Ġ$.', '?', 'ĊĊ', '!', '...', '. '}
        
        segment_data = []
        start = 0
        for i, token in enumerate(tokens):
            if token in delimiters:
                seg_toks = tokens[start:i+1]
                seg_attrs = attributions[start:i+1]
                strength = float((np.sum(np.asarray([np.abs(attr) for attr in seg_attrs]))))
                normalized_strength_1 = strength / np.sqrt(len(seg_attrs))
                consistency_numerator = float(np.abs(np.sum(np.asarray(seg_attrs))))
                consistency_denominator = float((np.sum(np.asarray([np.abs(attr) for attr in seg_attrs]))))
                consistency = consistency_numerator / consistency_denominator
                seg_data = {
                    'tokens': seg_toks,
                    'strength': strength,
                    'normalized_strength_1': normalized_strength_1,
                    'consistency': consistency
                }
                segment_data.append(seg_data)
                start = i
                
        return segment_data
    
    def get_segment_attribution_data(self, int_grad_result: dict):
        segment_data = self.segment_strengths_and_consistencies(int_grad_result)
        segment_data['normalized_strenth_2'] = self.normalize_segment_strengths(segment_data)
        for i, d in enumerate(segment_data):
            d['orig_idx'] = i
        return segment_data
    
    def filter_seg_attr_data(self, sorted_segment_data, tau, beta):
        k_star = 0
        total_normalized_strength = 0
        for item in sorted_segment_data:
            total_normalized_strength += item['normalzed_strength_2']
            k_star += 1
            if total_normalized_strength > tau:
                break
        top_segments = sorted_segment_data[:k_star]
        
        important_segments = []
        for item in top_segments:
            if item['consistency'] < beta:
                important_segments.append(item)
                
        return important_segments
        
    
    def get_important_segments_orig_order(self, segment_attribution_data, tau, beta):
        
        seg_attr_data_sorted_by_strength = sorted(
            segment_attribution_data,
            key=lambda d: d['normalzed_strength_2'],
            reverse=True
        )
        
        seg_attr_filtered = self.filter_seg_attr_data(
            seg_attr_data_sorted_by_strength,
            tau,
            beta
        )
        
        important_segments_orig_order = sorted(
            seg_attr_filtered,
            key=lambda d: d['orig_idx'],
            reverse=False
        )
        
        return important_segments_orig_order
        
    def run_int_grad_attr(self, data, prune: bool):
        # Extract the full_cot minus the answer
        full_cot = data['full_cot']
        cot_minus_answer = self.extract_cot_minus_answer(full_cot)
        
        # Get the target token id; first token of answer
        target_token_id = self.get_target_token_id(full_cot, cot_minus_answer)
        
        # Calculate the integradient gradients
        print("Calculating Integrated Gradients...")
        int_grad_result = integrated_gradients(
            model=self.model,
            input_text=cot_minus_answer,
            target_token_id=target_token_id
        )
        
        segment_attribution_data = self.get_segment_attribution_data(int_grad_result)
        
        if prune:
            
            important_segments_orig_order = self.get_important_segments_orig_order(
                segment_attribution_data,
                TAU,
                BETA
            )
            
            return important_segments_orig_order
        
        return segment_attribution_data
        
    def run_experiment(self):
        print("Loading dataset...")
        dataset = load_dataset(INPUT_FILE)
        print(f"Loaded {len(dataset)} problems.")
        
        # Open output file
        with open(OUTPUT_FILE, 'w') as f_out:
            for i, data in enumerate(dataset):
                result = self.run_int_grad_attr(data)
                
                f_out.write(json.dumps(result), "\n")
                f_out.flush()

if __name__ == "__main__":
    
    # Output: Save token attributions to a json file
    iga_experiment = IGAExperiment()
    iga_experiment.run_experiment(prune=PRUNE)
