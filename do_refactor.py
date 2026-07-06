import os
import re

def main():
    with open("symplectic_forecaster.py", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # We will find the start and end of each class/function
    # by looking for "class " or "def " at the root level (no indent)
    blocks = []
    current_block = []
    current_name = None
    current_type = None
    
    for line in lines:
        match = re.match(r'^(class|def)\s+([A-Za-z0-9_]+)', line)
        if match:
            # save previous block
            if current_block:
                blocks.append({
                    "type": current_type,
                    "name": current_name,
                    "lines": current_block
                })
            current_type = match.group(1)
            current_name = match.group(2)
            current_block = [line]
        elif line.startswith("if __name__ == "):
            if current_block:
                blocks.append({
                    "type": current_type,
                    "name": current_name,
                    "lines": current_block
                })
            current_type = "main_block"
            current_name = "main"
            current_block = [line]
        elif re.match(r'^[A-Z0-9_]+\s*=', line) and not line.startswith(" "):
            # Global variables
            if current_block:
                blocks.append({
                    "type": current_type,
                    "name": current_name,
                    "lines": current_block
                })
            current_type = "global_var"
            current_name = line.split('=')[0].strip()
            current_block = [line]
        else:
            if current_block:
                current_block.append(line)
            else:
                current_block = [line] # Imports and header
                current_type = "header"
                current_name = "header"
                
    if current_block:
        blocks.append({
            "type": current_type,
            "name": current_name,
            "lines": current_block
        })
        
    print(f"Found {len(blocks)} blocks:")
    for b in blocks:
        name = b['name']
        print(f"  {b['type']} {name} ({len(b['lines'])} lines)")

if __name__ == "__main__":
    main()
