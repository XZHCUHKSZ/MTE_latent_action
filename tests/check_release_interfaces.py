"""Small CPU interfaces and create-only/freeze barriers; no formal experiment."""
import json,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def check():
    import numpy as np
    from experiments.correspondence_controls import mapping,transform
    from evaluation.frozen_alignment import score
    rng=np.random.default_rng(43);x=rng.normal(size=(8,9,3));valid=np.ones((8,8),bool)
    donor,meta=mapping(x,valid,4,0,8,7)
    assert np.all((donor<4)==(np.arange(8)[:,None]<4)) and np.all(donor!=np.arange(8)[:,None])
    a=rng.normal(size=(8,8,3,8,4,2));b=rng.normal(size=a.shape);aa,bb=transform(a,b,donor,'misaligned_graph')
    assert np.array_equal(aa-bb,(a-b)[donor,np.arange(8)[None,:]])
    masks=np.array([[0,*[(v>>k)&1 for k in range(3)]] for v in range(8)])
    z=rng.normal(size=(8,9,4));tr=np.array([[e,t] for e in range(4) for t in [1,2,3]]);dv=tr.copy();dv[:,0]+=4
    rows=score(z,a,b,masks,np.eye(8),np.ones(2),tr,dv)
    assert len(rows)==36 and all(np.isfinite(r['skill']) for r in rows)
    assert sorted({r['shift'] for r in rows})==[-1,0,1]
    with tempfile.TemporaryDirectory() as d:
        d=Path(d);run=d/'run'
        base=[sys.executable,'-X','utf8','-B','-m','experiments.coupled_frozen']
        cmd=base+['init','--run-dir',str(run),'--data-root',str(d/'data'),'--teacher-root',str(d/'teachers')]
        def call(args):return subprocess.run(args,cwd=ROOT,capture_output=True,text=True,encoding="utf-8")
        first=call(cmd);assert first.returncode==0,first.stderr
        assert call(cmd).returncode!=0
        p=run/'protocol.json';p.write_text(p.read_text()+' ')
        rejected=call(base+['seal-data','--run-dir',str(run)])
        assert rejected.returncode!=0 and 'Frozen artifact changed' in rejected.stderr
    return dict(passed=True,correspondence='split-local derangement and intact edge transport',alignment_rows=36,barriers=['create-only','source/protocol mutation rejected'])
if __name__=='__main__':print(json.dumps(check()))
