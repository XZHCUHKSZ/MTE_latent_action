import os,json,time
from pathlib import Path

# Final implementation source: mte_observation_only_2026_09_09/manager.py:30

def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    for attempt in range(8):
        try:
            os.replace(temp,path)
            return
        except PermissionError:
            if attempt==7: raise
            time.sleep(.1*(attempt+1))
