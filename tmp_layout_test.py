from src.tvr_service.generator import build_layout_from_source, parse_layout_text, compile_layout

layout = build_layout_from_source('Default.tvr2', [5562, 5566])
print('LINES:\n' + layout.to_text())

edited_lines = []
for line in layout.to_lines():
    if line.startswith('#base_short'):
        edited_lines.append(f"{line}: 2")
    else:
        edited_lines.append(line)
edited_text = "\n".join(edited_lines)

edits = parse_layout_text(layout, edited_text)
print('OVERRIDES', edits.relative_overrides)
compiled = compile_layout(layout, edits, {5562: 1, 5566: 5})
print('COMPILED base', compiled.base_assignment)
print('COMPILED overrides', compiled.stroka_overrides)
