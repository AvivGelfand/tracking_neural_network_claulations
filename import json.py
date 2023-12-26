import json
import re

def get_imports_from_notebook(notebook_path):
    with open(notebook_path, 'r') as f:
        data = json.load(f)

    imports = set()
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            for line in cell['source']:
                match = re.match(r'import (\S+)|from (\S+) import', line)
                if match:
                    imports.add(match.group(1) or match.group(2))

    return list(imports)

# Use the function
notebook_path = 'main3.ipynb'  # replace with your notebook path
imports = get_imports_from_notebook(notebook_path)
print(imports)
