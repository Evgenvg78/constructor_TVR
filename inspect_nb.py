import nbformat
from pathlib import Path

nb = nbformat.read(Path('docs/strategy_mask_demo.ipynb').open('r', encoding='utf-8'), as_version=4)
for idx, cell in enumerate(nb.cells):
    first_line = cell.source.split('\n', 1)[0]
    print(f"{idx}: {cell.cell_type} -> {first_line}")
