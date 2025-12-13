"""
handoff.py
"""

import torch
from vllm import LLM, SamplingParams
from _src.text_utils import split_into_steps, check_correctness, apply_chat_template

class HandoffEngine:
    def __init__(self, model_path: str, gpus: int = 1):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=gpus,
            trust_remote_code=True,
            max_model_len=32768
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            n=5,
            temperature=0.7,
            top_p=0.95,
            max_tokens=12288
        )

    def compute_hsr(
        self,
        problem: str,
        ground_truth: str,
        senior_trace: str
    ) -> list[float]:
        """
        Calculates HSR(k) for every step k in the senior trace.
        """
        steps = split_into_steps(senior_trace)
        hsr_curve = []

        # We process all steps. For efficiency, you might batch this.
        # This loop logic prepares prompts for ALL steps to run in one vLLM batch.
        prompts = []

        # Build prefix for each step k
        # Prefix_k includes step 0 to k (inclusive)
        current_prefix = ""
        for k, step in enumerate(steps):
            current_prefix += step + "\n"

            # Format as a chat: "User: problem", "Assistant: prefix..."
            # Note: We need the assistant to continue FROM current_prefix.
            # vLLM doesn't support 'partial assistant' in chat template easily
            # without manual formatting.
            # Strategy: Format prompt manually or use `continue_final_message` if supported.

            # Simplified Strategy: Construct raw prompt string
            # (Assumes Qwen/ChatML format)
            # <im_start>user\n{problem}<|im_end|>\n<|im_start|>assistant\n{current_prefix}

            # Use tokenizer to build base, then append prefix
            base_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": problem}],
                tokenize=False,
                add_generation_prompt=True
            )
            full_prompt = base_prompt + current_prefix
            prompts.append(full_prompt)

        # Generate completions for all steps in parallel
        outputs = self.llm.generate(prompts, self.sampling_params)

        # Calculate Reward (HSR) for each step
        for k, output in enumerate(outputs):
            success_count = 0
            for sample in output.outputs:
                # Full trace = prefix + completion
                # We check if the completion *arrives* at the correct answser.
                full_completion = sample.text
                if check_correctness(problem, full_completion, ground_truth):
                    success_count += 1

            hsr_k = success_count / len(output.outputs)
            hsr_curve.append({
                "step_idx": k,
                "step_text": steps[k],
                "hsr": hsr_k
            })

        return hsr_curve