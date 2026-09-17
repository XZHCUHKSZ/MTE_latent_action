"""Final coordinate pilot, explicit configuration instead of dated job folders."""
import time,atexit,gc
from pathlib import Path
import numpy as np
import torch
from closed_loop_lam_v1 import common as C
from closed_loop_lam_v1.unified_models import lattice_operators
from mte.coordinate_controls import factory,forward
from evaluation.probes import reader
from utils.artifacts import rich,endpoints
from utils.access import guard
from utils.atomic import atomic_json

def run(j,p,out):
    s=j['suite'];out=Path(out);out.mkdir(exist_ok=False)
    start=time.time();torch.set_num_threads(1)
    audit=guard(out,[Path(s[k]) for k in ['train','dev']]+[endpoints(s)],True)
    atexit.register(lambda:atomic_json(out/'access.json',audit))
    def progress(**kw):atomic_json(out/'progress.json',dict(elapsed=time.time()-start,**kw))
    progress(phase='loading',update=0)
    x,m,n,a,b,masks,adj,valid=rich(s)
    offset=1 if s['env']=='mpe' else 0;t=np.arange(valid.shape[1])+offset
    support=valid & (t+3<=m.shape[1])[None] & m[:,np.minimum(t+2,m.shape[1]-1)]
    tr=np.argwhere(support[:n]);dv=np.argwhere(support[n:]);dv[:,0]+=n
    rng=np.random.default_rng(91832+s['seed']);tr=tr[rng.choice(len(tr),min(p['train_max'],len(tr)),replace=False)]
    rng=np.random.default_rng(58710+s['seed']);dv=dv[rng.choice(len(dv),min(p['dev_max'],len(dv)),replace=False)]
    if j['model']=='mif':
        def table(ix):
            e,t=ix.T;return np.concatenate([a[e,t],a[e,t]-b[e,t]],-1)
        yt,yv=table(tr),table(dv);edge_dim=a.shape[-1]
    else:
        del a,b
        with np.load(endpoints(s),allow_pickle=False) as f:a,b=f['actual'],f['reference']
        def table(ix):
            e,t=ix.T;return a[e,t,2]-b[e,t,2]
        yt,yv=table(tr),table(dv);edge_dim=yt.shape[-1]
    del a,b,x;gc.collect()
    mu,sd=C.fit_mean_std(yt.reshape(-1,yt.shape[-1]));yt=(yt-mu)/sd;yv=(yv-mu)/sd
    y=torch.tensor(yt,device='cuda');v=torch.tensor(yv,device='cuda')
    M=lattice_operators(torch.tensor(masks))['mobius'].cuda()
    def edge(q):return q[...,edge_dim:] if j['model']=='mif' else q
    def coeff(q):return torch.einsum('cd,bhdek->bhcek',M,edge(q)) if j['model']=='mif' else torch.einsum('cd,bdk->bck',M,q)
    C.seed_all(s['seed']);net=factory(j['model'],j['variant'],y.shape[-1],masks,adj)
    initial=[v.detach().clone() for v in net.parameters()]
    # The paired controls must have exactly identical parameter initialization.
    C.seed_all(s['seed']);reference=factory(j['model'],'mobius',y.shape[-1],masks,adj)
    assert all(torch.equal(q,r) for q,r in zip(initial,reference.parameters()))
    parameters=sum(v.numel() for v in net.parameters());assert parameters==sum(v.numel() for v in reference.parameters())
    del reference,initial
    opt=torch.optim.Adam(net.parameters(),lr=p['learning_rate']);rng=np.random.default_rng(s['seed']+17)
    torch.cuda.synchronize();training_start=time.time()
    for step in range(p['updates']):
        ix=rng.choice(len(y),p['batch'],replace=False);z,r=forward(net,y[ix],j['model'])
        loss=(r-y[ix]).square().mean()+C._variance_floor(z)+.02*C._offdiag_cov(z)
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
        if step%40==0 or step+1==p['updates']:progress(phase='training',update=step+1,total=p['updates'],loss=float(loss.detach()))
    torch.cuda.synchronize();training_seconds=time.time()-training_start
    active=sum(q.numel() for q in net.parameters() if q.grad is not None)
    net.eval()
    def infer(q):
        zs=[];rs=[]
        with torch.no_grad():
            for chunk in q.split(32):
                z,r=forward(net,chunk,j['model']);zs.append(z.cpu().numpy());rs.append(r.cpu().numpy())
        return np.concatenate(zs),np.concatenate(rs)
    tz,_=infer(y);dz,r=infer(v);ct=coeff(y).cpu().numpy();cv=coeff(v).cpu().numpy()
    cr=coeff(torch.tensor(r,device='cuda')).cpu().numpy()
    lr=reader(tz.astype(float),ct.reshape(len(ct),-1).astype(float),dz.astype(float)).reshape(cv.shape)
    rows=[];axis=2 if j['model']=='mif' else 1
    def add(kind,group,pred,truth,mean):
        mse=float(np.mean((pred-truth)**2));den=float(np.mean((truth-mean)**2))
        rows.append(dict(kind=kind,group=group,mse=mse,baseline_mse=den,skill=1-mse/den if den>1e-12 else None))
    for view,pred in [('native',cr),('linear',lr)]:
        for k,mask in enumerate(masks):
            add(view,','.join(map(str,np.flatnonzero(mask))) or 'empty',np.take(pred,k,axis),np.take(cv,k,axis),np.take(ct.mean(0),k,axis-1))
    add('edge','all',edge(torch.tensor(r)).numpy(),edge(torch.tensor(yv)).numpy(),edge(torch.tensor(yt)).numpy().mean(0))
    torch.save(dict(state_dict=net.cpu().state_dict(),job=j,mean=mu,std=sd,masks=masks,adjacency=adj),out/'representation.pt')
    np.savez_compressed(out/'latents.npz',train=tz,dev=dz,train_indices=tr,dev_indices=dv)
    assert not audit['violations'];atomic_json(out/'access.json',audit)
    atomic_json(out/'result.json',dict(complete=True,job=j,rows=rows,parameters=parameters,active_gradient_parameters=active,
        training_seconds=training_seconds,elapsed_seconds=time.time()-start,init_equal=True,native_action_labels_read=0,simulator_queries=0))
    progress(phase='complete',update=p['updates'])

