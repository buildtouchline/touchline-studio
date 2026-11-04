"""Manual syntax/notebook/link checks. This is not CI or a GPU execution test."""
import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote

root=Path(__file__).resolve().parents[1]
errors=[];python_files=0;notebook_cells=0;links=0
for path in root.rglob('*.py'):
    if any(p in ('.venv','.test-temp','__pycache__','runs') for p in path.parts):continue
    ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path));python_files+=1
for path in (root/'notebooks').glob('*.ipynb'):
    notebook=json.loads(path.read_text(encoding='utf-8'))
    assert notebook['nbformat']==4
    for i,cell in enumerate(notebook['cells']):
        if cell['cell_type']=='code':
            source=cell['source'] if isinstance(cell['source'],str) else ''.join(cell['source'])
            ast.parse(source,filename=f'{path.name}:cell{i}')
            assert cell['execution_count'] is None and not cell['outputs'], 'Do not ship fabricated execution evidence'
            notebook_cells+=1
for path in root.rglob('*.md'):
    if any(p in ('.venv','.test-temp','__pycache__','runs') for p in path.parts):continue
    for match in re.finditer(r'\[[^\]]*\]\(([^\s]+)(?:\s+"[^"]*")?\)',path.read_text(encoding='utf-8-sig')):
        target=unquote(match.group(1)).strip('<>')
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target) or target.startswith('#'):continue
        target=target.split('#',1)[0]
        if not target:continue
        links+=1
        if not (path.parent/target).exists():errors.append(f'{path.relative_to(root)} -> {target}')
assert not list(root.glob('.github/workflows/*')), 'No workflow files requested'
print(json.dumps({'python_files_parsed':python_files,'notebook_code_cells_parsed':notebook_cells,
    'local_markdown_links_checked':links,'broken_links':errors,'gpu_execution':'not run'},indent=2))
if errors:raise SystemExit(1)
