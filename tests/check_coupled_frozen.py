"""CPU grounding parity against the archived Coupled adapter; no simulator/GPU."""
import argparse,importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def worker(a):
 import numpy as np,torch
 if a.reference:
  spec=importlib.util.spec_from_file_location('coupled_archive_reference',a.reference);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 else:
  from training import coupled_cheetah_frozen as mod
 c={'device':'cpu','threads':1,'budget_episodes':8,'ground_updates':2,'qualification':True}
 out=Path(a.out);out.mkdir(exist_ok=False)
 r=mod.ground(c,202623500,Path(a.fixture)/'pre',Path(a.fixture)/'target_actions.npy',out,a.arm,lambda **kw:None)
 (out/'check.json').write_text(json.dumps(r),encoding='utf-8')

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--reference',type=Path);ap.add_argument('--worker',action='store_true');ap.add_argument('--fixture');ap.add_argument('--out');ap.add_argument('--arm');a=ap.parse_args()
 if a.worker:return worker(a)
 import numpy as np,torch
 from closed_loop_lam_v1.common import RecurrentPolicy
 from training.coupled_cheetah_frozen import arms
 assert len(arms())==19 and not any('Adapt' in s for s in arms())
 if not a.reference:ap.error('For independent extraction parity, pass the archived training/coupled_cheetah_method_adapter_r3.py')
 a.reference=a.reference.resolve();cases=[]
 with tempfile.TemporaryDirectory(prefix='coupled_parity_') as tmp:
  base=Path(tmp);pre=base/'pre';pre.mkdir();rng=np.random.default_rng(7788);torch.manual_seed(7788)
  np.savez_compressed(pre/'features.npz',x=rng.normal(size=(68,201,2,32)).astype('float32'))
  np.save(base/'target_actions.npy',rng.uniform(-1,1,(8,200,6)).astype('float32'))
  for name,dim in [('base',64),('edge_cara_mif',16)]:
   p=pre/name/'policy';p.mkdir(parents=True);net=RecurrentPolicy(64,dim)
   torch.save(dict(state_dict=net.state_dict(),obs_mean=np.zeros(64,'float32'),obs_std=np.ones(64,'float32'),z_dim=dim,frozen_before_grounding=True,native_action_labels_read=0),p/'policy.pt')
  for arm in ['base__Frozen','solo_edge_cara_mif__Frozen','aux_edge_cara_mif__Frozen','bc','idm']:
   dirs=[]
   for label in ['reference','published']:
    out=base/(arm+'_'+label);cmd=[sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'--worker','--fixture',str(base),'--out',str(out),'--arm',arm]
    if label=='reference':cmd+=['--reference',str(a.reference)]
    done=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8')
    if done.returncode:raise RuntimeError(done.stdout+done.stderr)
    dirs.append(out)
   checkpoints=[torch.load(p/'controller.pt',map_location='cpu',weights_only=False) for p in dirs]
   def compare(x,y):
    if isinstance(x,torch.Tensor):assert torch.equal(x,y)
    elif isinstance(x,np.ndarray):assert np.array_equal(x,y)
    elif isinstance(x,dict):
     assert x.keys()==y.keys()
     for k in x:compare(x[k],y[k])
    else:assert x==y,(x,y)
   compare(*checkpoints)
   if arm=='idm':compare(*(torch.load(p/'idm.pt',weights_only=False) for p in dirs))
   rs=[json.loads((p/'check.json').read_text()) for p in dirs];assert rs[0]==rs[1]
   cases.append({'arm':arm,'controller_tensors_exact':True,'diagnostics_exact':True,'updates':2,'batch':256})
   print('PASS '+arm,flush=True)
 print(json.dumps({'passed':True,'cases':cases,'scope':'Synthetic CPU extraction parity; not performance reproduction'}))
if __name__=='__main__':main()
