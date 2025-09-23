import nbformat
from pathlib import Path

nb = nbformat.read(Path('docs/strategy_mask_demo.ipynb').open('r', encoding='utf-8'), as_version=4)

indices = [6,10,11,13,15]
for idx in indices:
    cell = nb.cells[idx]
    print(f"--- cell {idx} ---")
    print(cell.source)
    print()
