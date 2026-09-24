import os,sys
from pathlib import Path
import numpy as np

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/core.py:27

def guard(out,allowed,pretraining=True):
    allowed={Path(p).resolve() for p in allowed};audit={'numeric_reads':[],'violations':[],'pretraining':pretraining}
    # CPython places this exact, often nonexistent standard-library archive on
    # sys.path. Importlib probes it; it is not an experimental numeric artifact.
    runtime_zip=(Path(sys.base_prefix)/f'python{sys.version_info.major}{sys.version_info.minor}.zip').resolve()
    audit['missing_stdlib_probes']=[]
    def hook(event,args):
        if pretraining and event=='import' and args[0].startswith(('gymnasium','mujoco','pettingzoo','simglucose')):
            audit['violations'].append(args[0]);raise RuntimeError('Pretraining cannot import simulators')
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.suffix.lower() not in ('.npz','.npy','.pt','.pth','.pkl','.pickle','.h5','.hdf5','.zip'):return
        writing=bool(args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT))
        if not writing and p==runtime_zip and not p.exists():
            audit['missing_stdlib_probes'].append(str(p));return
        if not writing:audit['numeric_reads'].append(str(p))
        if not p.is_relative_to(out.resolve()) and (writing or p not in allowed):
            audit['violations'].append(str(p));raise RuntimeError('Artifact outside stage whitelist: '+str(p))
    sys.addaudithook(hook);return audit

# Final implementation source: mamujoco_observation_only_v2_2026_09_10/core.py:47

def read_rows(source,member,ids):
    """Read only selected episode rows from one NPY member; never load the whole archive."""
    import zipfile
    with zipfile.ZipFile(source) as z, z.open(member+'.npy') as f:
        version=np.lib.format.read_magic(f)
        shape,order,dtype=(np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0)(f)
        assert not order and not dtype.hasobject
        offset=f.tell();size=int(np.prod(shape[1:]))*dtype.itemsize
        values=[]
        for i in ids:
            assert 0<=i<shape[0];f.seek(offset+int(i)*size)
            raw=f.read(size);assert len(raw)==size
            values.append(np.frombuffer(raw,dtype=dtype).reshape(shape[1:]).copy())
    return np.stack(values)
