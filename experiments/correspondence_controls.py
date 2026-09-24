"""Split-local observation-based permutations; never accepts action labels."""
import numpy as np
from scipy.optimize import linear_sum_assignment

def mapping(x,valid,n,start,steps,seed):
    assert 0<n<len(x) and x.shape[0]==valid.shape[0]
    e=len(x);history=np.concatenate([x[:,np.maximum(np.arange(valid.shape[1])-1,0)],x[:,:-1]],-1)
    tr=history[:n][valid[:n]];mean=tr.mean(0);std=tr.std(0).clip(1e-6)
    hist=(history-mean)/std
    donor=np.broadcast_to(np.arange(e)[:,None],(e,steps)).copy()
    rng=np.random.default_rng(seed+400001);singletons=eligible=changed=0;distances=[]
    for t in range(steps):
        tt=t+start
        future=np.stack([valid[:,min(tt+h,valid.shape[1]-1)] & (tt+h<valid.shape[1]) for h in range(3)],-1)
        patterns=(future*np.array([1,2,4])).sum(-1)
        for lo,hi in ((0,n),(n,e)):
            for pattern in np.unique(patterns[lo:hi]):
                ids=np.arange(lo,hi)[patterns[lo:hi]==pattern]
                if not (pattern&1):continue
                eligible+=len(ids)
                if len(ids)<2:singletons+=len(ids);continue
                hh=hist[ids,tt].astype(np.float64)
                # Pairwise squared distances without a (episodes,episodes,features) allocation.
                norms=(hh*hh).sum(1);cost=np.maximum(norms[:,None]+norms[None,:]-2*hh@hh.T,0)/hh.shape[1]
                cost+=rng.uniform(0,1e-10,cost.shape);np.fill_diagonal(cost,np.inf)
                a,b=linear_sum_assignment(cost);assert np.all(a!=b)
                donor[ids[a],t]=ids[b];changed+=len(ids);distances.extend(cost[a,b].tolist())
    assert np.all((donor<n)==(np.arange(e)[:,None]<n))
    return donor,dict(eligible=eligible,changed=changed,singletons_unchanged=singletons,
        changed_fraction=changed/max(eligible,1),history_distance_quantiles=np.quantile(distances,[.5,.9,.99]).tolist() if distances else [],
        same_timestamp=True,split_preserved=True,validity_pattern_preserved=True,selection_uses='previous and current observations only')

def transform(a,b,donor,arm):
    assert a.shape==b.shape and a.shape[:2]==donor.shape
    tt=np.arange(a.shape[1])[None,:]
    if arm=='matched_graph':return a,b
    if arm=='unpaired_graph':return a,b[donor,tt]
    if arm=='misaligned_graph':return a[donor,tt],b[donor,tt]
    raise ValueError(arm)

def diagnostics(a,b,aa,bb,valid,n,start):
    v=valid[:,start:start+a.shape[1]];out={}
    for name,sl in [('train',slice(0,n)),('dev',slice(n,None))]:
        before=(a[sl]-b[sl])[v[sl]];after=(aa[sl]-bb[sl])[v[sl]]
        out[name]=dict(before_abs_quantiles=np.quantile(np.abs(before),[.5,.9,.99]).tolist(),
            after_abs_quantiles=np.quantile(np.abs(after),[.5,.9,.99]).tolist(),
            before_rms=float(np.sqrt(np.mean(before.astype(np.float64)**2))),
            after_rms=float(np.sqrt(np.mean(after.astype(np.float64)**2))))
    return out
