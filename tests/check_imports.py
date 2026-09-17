"""Import and global-name check. Does not construct environments or train."""
import importlib,symtable,builtins,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def run():
    errors=[];count=0
    for folder in ['utils','mte','training','evaluation','environments','visual','experiments']:
        for f in (ROOT/folder).glob('*.py'):
            if f.name=='__init__.py':continue
            name=folder+'.'+f.stem;count+=1
            try:mod=importlib.import_module(name)
            except Exception as e:errors.append([name,'import',str(e)]);continue
            def check(t):
                if t.get_type()=='function':
                    for s in t.get_symbols():
                        if s.is_referenced() and s.is_global() and s.get_name() not in vars(mod) and not hasattr(builtins,s.get_name()):errors.append([name,t.get_name(),s.get_name()])
                for c in t.get_children():check(c)
            check(symtable.symtable(f.read_text(encoding='utf8'),str(f),'exec'))
    return dict(modules=count,errors=errors)

if __name__=='__main__':
    r=run();print(json.dumps(r,indent=2));sys.exit(bool(r['errors']))
