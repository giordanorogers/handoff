import re

def extract_answer(text: str) -> str:
    """
    Extracts the answer from a reasoning trace.
    Prioritizes \boxed{} content, which is standard for MATH/Qwen.
    """
    # 1. Look for LaTeX boxed answer: \boxed{...}
    # We use a greedy search for the last \boxed because models often 
    # output intermediate boxed steps (though Qwen usually boxes only the final).
    boxed_matches = re.findall(r'\\boxed\{(.*?)\}', text)
    if boxed_matches:
        return boxed_matches[-1].strip()
    
    # 2. Fallback: GSM8K style ####
    if "####" in text:
        return text.split("####")[1].strip()
        
    return "[No Answer Found]"

def normalize_math_answer(answ: str) -> str:
    """
    Normalizes math strings for comparison.
    e.g., "1/2" == "0.5", "x+y" == "y+x" (simple cases).
    """
    # Remove text, whitespace, currency, and latex formatting
    answ = answ.replace('\\', '').replace('$', '').replace('%', '')
    answ = answ.replace('text', '').replace(' ', '')
    return answ

def split_into_steps(text: str) -> list[str]:
    """
    Splits reasoning into steps.
    Math reasoning often uses newlines as natural delimiters.
    """
    # Split by newline first
    lines = text.split('\n')
    steps = [line.strip() for line in lines if line.strip()]
    return steps

def apply_chat_template(tokenizer, query: str) -> str:
    messages = [{"role": "user", "content": query}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# src/text_utils.py
import re
import sympy
from sympy.parsing.latex import parse_latex

def extract_boxed_content(text: str) -> str:
    """
    Extracts content from \boxed{...} using a stack to handle nested braces.
    """
    if "\\boxed{" not in text:
        return None
    # Find the last occurrence of \boxed{
    start_idx = text.rfind("\\boxed{")
    if start_idx == -1:
        return None
    
    content_start = start_idx + 7 # len("\\boxed{")
    balance = 1
    content_end = content_start
    
    while content_end < len(text) and balance > 0:
        char = text[content_end]
        if char == '{':
            balance += 1
        elif char == '}':
            balance -= 1
        content_end += 1
        
    if balance == 0:
        return text[content_start : content_end - 1]
    return None

def extract_option_value(problem_text: str, option_letter: str) -> str:
    """
    If the model predicts 'A', tries to find '(A) value' in the problem text.
    """
    pattern = f"\\({option_letter}\\)\\s*(.*?)(?=\\([A-E]\\)|$)"
    match = re.search(pattern, problem_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def normalize_string(s: str) -> str:
    """Aggressive string normalization."""
    if not s: return ""
    s = s.replace(r'\dfrac', r'\frac').replace(r'\tfrac', r'\frac')
    s = s.replace(r'\left', '').replace(r'\right', '')
    # Add .replace(',', '') here:
    s = s.replace('$', '').replace(' ', '').replace(',', '')
    return s

def check_correctness(problem_text: str, prediction: str, ground_truth: str) -> bool:
    """
    Robust correctness check handling LaTeX, options, and symbolic math.
    """
    pred_raw = extract_boxed_content(prediction)
    gt_raw = extract_boxed_content(ground_truth)
    
    if not pred_raw: 
        # Fallback for when model forgets boxing but outputs answer at end
        # (Simplified fallback)
        pred_raw = prediction.split('\n')[-1] 
    if not gt_raw:
        gt_raw = ground_truth

    # 1. Handle MCQ Options (Model says 'A', GT is '29')
    if pred_raw in ['A', 'B', 'C', 'D', 'E'] and len(gt_raw) > 1:
        mapped_val = extract_option_value(problem_text, pred_raw)
        if mapped_val:
            pred_raw = mapped_val

    # 2. String Normalization Check
    pred_norm = normalize_string(pred_raw)
    gt_norm = normalize_string(gt_raw)
    if pred_norm == gt_norm:
        return True
        
    # 3. SymPy Symbolic Check (Calculus/Algebra equivalence)
    try:
        # Very basic latex-to-sympy (requires strict latex)
        # Often easier to wrap in simplify(pred - gt) == 0
        # If latex parsing fails, we skip
        expr_pred = parse_latex(pred_raw)
        expr_gt = parse_latex(gt_raw)
        if sympy.simplify(expr_pred - expr_gt) == 0:
            return True
    except:
        pass

    return False

def split_into_steps(text: str) -> list[str]:
    """Splits reasoning into steps (sentences/newlines)."""
    # Simple split by newline, can be enhanced with NLTK sentence tokenizer
    lines = text.split('\n')
    steps = [line.strip() for line in lines if line.strip()]
    return steps