import re
import json

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