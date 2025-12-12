import json
import re

notebook_path = r"C:\Users\nandi\Desktop\test\Agent Trading.ipynb"

try:
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"Notebook loaded. Cells: {len(nb['cells'])}")

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            # Find classes
            classes = re.findall(r"class\s+(\w+)", source)
            if classes:
                print(f"Cell {i}: Classes found: {classes}")
            
            # Find imports
            imports = re.findall(r"import\s+(\w+)", source)
            if imports:
                print(f"Cell {i}: Imports found: {imports}")
                
            # Extract RL and NeuroEvolution
            if 80 <= i <= 130:
                print(f"Cell {i}: Extracting code...")
                with open("extracted_agents_v14.py", "a", encoding="utf-8") as out:
                    out.write(f"# Cell {i}\n")
                    out.write(source)
                    out.write("\n\n")

except Exception as e:
    print(f"Error: {e}")
