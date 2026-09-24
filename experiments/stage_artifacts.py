"""Scientific stage APIs retained for the manuscript experiment; no background manager.

See docs/PAPER_CODE_MAP.md for the manuscript experiment mapping.
"""

import hashlib

import json

from pathlib import Path

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def location(out,job):
    phase,suite,seed,arm,kind=job
    return out/phase/suite/f'seed{seed}'/(kind+'_'+arm)

def verify_sources(out):
    for f,h in read(out/'source_manifest.json').items():assert digest(f)==h,f

def progress(path):
    try:return read(path)
    except (FileNotFoundError,PermissionError,json.JSONDecodeError):return {'telemetry_unavailable':True}
