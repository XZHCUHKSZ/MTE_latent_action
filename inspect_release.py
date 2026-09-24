"""Check source/configuration syntax and the code-only distribution policy."""
import argparse,ast,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FOLDERS=['closed_loop_lam_v1','configs','environments','evaluation','experiments','mte','training','utils','visual','tests','third_party']
def verify():
    counts={'python':0,'configs':0}
    for folder in FOLDERS:
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:continue
            if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'));counts['python']+=1
            elif p.suffix=='.json':
                assert folder=='configs',str(p)
                json.loads(p.read_text(encoding='utf-8-sig'));counts['configs']+=1
            else:assert p.name in ['LICENSE','README.md'],str(p)
    for folder in ['results','provenance']:
        assert not any(p.is_file() for p in (ROOT/folder).rglob('*')),folder+' must not be distributed'
    return dict(passed=True,**counts,scope='Code and configuration only; no experiment execution')
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--verify',action='store_true');args=a.parse_args()
    if args.verify:print(json.dumps(verify()))
if __name__=='__main__':main()
