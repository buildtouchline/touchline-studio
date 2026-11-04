"""Validate synchronized planning exports and local links without providers."""
import argparse,csv,json,re
from pathlib import Path
from urllib.parse import unquote

def check(root):
    root=Path(root).resolve();parents=json.loads((root/'docs/tickets/backlog.json').read_text(encoding='utf-8'));subtasks=json.loads((root/'docs/tickets/subtasks.json').read_text(encoding='utf-8'))
    rows=parents['tickets']+subtasks['subtasks'];ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate ticket ID')
    lookup={r['id']:r for r in rows};specs={re.match(r'TL-S\d+',p.stem).group() for p in (root/'docs/specs').glob('TL-S*.md')}
    for row in rows:
        if row['status'] not in {'Planned','Partially done for demo','Done','Blocked','In progress'}:raise ValueError('Unexpected status')
        for dep in row.get('dependencies',[]):
            if dep not in lookup:raise ValueError('Missing dependency '+dep)
        for spec in row.get('specs',[]):
            if spec not in specs:raise ValueError('Missing spec '+spec)
        if 'parent_id' in row:
            if row['parent_id'] not in lookup or row['id'] not in lookup[row['parent_id']].get('subtasks',[]):raise ValueError('Subtask-parent mismatch '+row['id'])
        else:
            for child in row.get('subtasks',[]):
                if child not in lookup or lookup[child].get('parent_id')!=row['id']:raise ValueError('Parent-subtask mismatch '+row['id'])
    visited=set();active=set()
    def visit(identifier):
        if identifier in active:raise ValueError('Dependency cycle')
        if identifier in visited:return
        active.add(identifier)
        for dep in lookup[identifier].get('dependencies',[]):visit(dep)
        active.remove(identifier);visited.add(identifier)
    for identifier in lookup:visit(identifier)
    for name,key,obj in [('backlog','tickets',parents),('subtasks','subtasks',subtasks)]:
        with (root/f'docs/tickets/{name}.csv').open(encoding='utf-8',newline='') as stream:
            reader=csv.DictReader(stream);actual=list(reader);fields=reader.fieldnames
        expected=[]
        for row in obj[key]:expected.append({k:json.dumps(row[k],ensure_ascii=False) if isinstance(row.get(k),(list,dict)) else '' if row.get(k) is None else str(row.get(k,'')) for k in fields})
        if actual!=expected:raise ValueError('CSV/JSON mismatch '+name)
    links=0;broken=[]
    for path in root.rglob('*.md'):
        if '.git' in path.parts:continue
        for match in re.finditer(r'\[[^\]]*\]\(([^\s]+)(?:\s+"[^"]*")?\)',path.read_text(encoding='utf-8-sig')):
            target=unquote(match.group(1)).strip('<>')
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target) or target.startswith('#'):continue
            target=target.split('#',1)[0]
            if not target:continue
            links+=1
            if not (path.parent/target).exists():broken.append(str(path.relative_to(root))+' -> '+target)
    if broken:raise ValueError('Broken local ticket links: '+json.dumps(broken))
    return {'status':'passed','parents':len(parents['tickets']),'subtasks':len(subtasks['subtasks']),'local_links':links,'dependencies_acyclic':True,'csv_json_match':True}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();print(json.dumps(check(args.root)))
