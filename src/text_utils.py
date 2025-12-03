"""
text_utils.py

Handles text parsing, formatting, and processing.
"""


import re

def extract_answer(completion: str) -> str:
    """
    Extracts the numerical answer from GSM8K predictions.
    Standard format is usually '#### <number>'
    """
    if "####" in completion:
        return completion.split("####")[1].strip()
    
    # Fallback: find the last number in the text (heuristic)
    # Matches integers or decimals, ignores commas
    numbers = re.findall(r'-?\d+(?:\.\d+)?', completion.replace(',', ''))
    if numbers:
        return numbers[-1]
    return "[No Answer Found]"

def check_correctness(prediction: str, ground_truth: str) -> bool:
    """
    Compares extracted prediction against ground truth.
    Handles distinct formats (e.g., '14' vs '14.0')
    """
    pred = extract_answer(prediction)
    gt = extract_answer(ground_truth)

    try:
        return float(pred) == float(gt)
    except ValueError:
        return pred == gt
    
def split_into_steps(text: str) -> list[str]:
    """
    Splits a reasoning chain into logical steps.

    Strategy:
    1. Macro-split by newlines.
    2. Micro-split by sentence terminators (.?!) followed by whitespace.
       This protects decimal numbers (e.g., "2.5") which are NOT followed by whitespace.
    """
    # Split by newline
    raw_lines = text.split('\n')

    final_steps = []

    # Split by sentence
    sentence_split_pattern = r'(?<=[.!?])\s+'
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        sentences = re.split(sentence_split_pattern, line)

        for sent in sentences:
            sent = sent.strip()
            if sent:
                final_steps.append(sent)

    return final_steps

def apply_chat_template(tokenizer, query: str, history: list = None) -> str:
    """
    Applies the specific model's chat template. (e.g., ChatML for Qwen)
    """
    messages = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})

    # tokenize=False returns the raw prompt string
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt
