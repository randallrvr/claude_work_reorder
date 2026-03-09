import json
import os
import sys
import traceback
import matplotlib
matplotlib.use('Agg')  # Set matplotlib to non-interactive backend

# Change to the working directory where shader files are located
work_dir = '/sessions/ecstatic-practical-gates/mnt/claude_cowork_reorder/'
os.chdir(work_dir)
print(f"Working directory set to: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')[:10]}")
print("\n" + "="*80)

# Load the notebook
notebook_path = 'work_reorder.ipynb'
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Extract all code cells
code_cells = []
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        code_cells.append((i, source))

print(f"Found {len(code_cells)} code cells to execute\n")

# Create a shared globals dictionary for cell execution
cell_globals = {
    '__name__': '__main__',
    '__builtins__': __builtins__,
}

# Execute cells sequentially
passed = 0
failed = 0
errors = []

for cell_index, source in code_cells:
    first_line = source.split('\n')[0][:70]
    print(f"Executing cell {cell_index}: {first_line}")
    
    try:
        exec(source, cell_globals)
        print(f"  ✓ Passed\n")
        passed += 1
    except Exception as e:
        print(f"  ✗ Failed with error:")
        print(f"    {type(e).__name__}: {str(e)[:150]}")
        print(f"    Traceback:")
        tb_lines = traceback.format_exc().split('\n')
        for line in tb_lines[-10:]:
            if line.strip():
                print(f"      {line}")
        print()
        failed += 1
        errors.append((cell_index, first_line, str(e)))

# Print summary
print("="*80)
print(f"\nSUMMARY")
print(f"  Total cells executed: {len(code_cells)}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")

if errors:
    print(f"\nFailed cells:")
    for idx, first_line, error in errors:
        print(f"  - Cell {idx}: {first_line}")
        print(f"    Error: {error[:100]}")

sys.exit(0 if failed == 0 else 1)
