"""Artifact integrity barrier; action labels must remain in a separate export."""
import json,hashlib
from pathlib import Path

def verify(record):
    p=Path(record);r=json.loads(p.read_text(encoding='utf8'))
    if not r.get('all_sources_frozen') or r.get('training_seeds')!=list(range(908711,908716)):
        raise ValueError('Require completed freeze records for all five visual seeds')
    if not r.get('artifacts'):raise ValueError('Empty freeze record')
    for item in r['artifacts']:
        f=Path(item['path'])
        if not f.is_absolute():f=p.parent/f
        actual=hashlib.sha256(f.read_bytes()).hexdigest()
        if actual!=item['digest']:raise ValueError('Artifact changed after freeze: '+str(f))
    return r
