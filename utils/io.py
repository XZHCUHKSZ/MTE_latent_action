import json,hashlib
from pathlib import Path

# Final implementation source: observation_action_probe_2026_09_11/run.py:28

def read(p): return json.loads(p.read_text(encoding='utf-8'))

# Final implementation source: observation_action_probe_2026_09_11/run.py:29

def write(p, obj): p.write_text(json.dumps(obj, indent=2, ensure_ascii=False),encoding='utf-8')

# Final implementation source: observation_action_probe_2026_09_11/run.py:30

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
