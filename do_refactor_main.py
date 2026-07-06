import re

with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Remove the AI ENGINE init block
pattern = r"    shared_nlp = None.*?    def run_symbol\(sym: str\):"
replacement = r"    shared_nlp = None\n\n    def run_symbol(sym: str):"
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open("main.py", "w", encoding="utf-8") as f:
    f.write(content)
