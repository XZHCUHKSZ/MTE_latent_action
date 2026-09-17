import numpy as np

# Final implementation source: observation_action_probe_2026_09_11/run.py:43

def ridge_probe(x,y,vx,vy,tx,lambdas):
    mean=x.mean(0);std=x.std(0).clip(1e-6);ym=y.mean(0)
    xx=(x-mean)/std;vv=(vx-mean)/std;tt=(tx-mean)/std
    cov=xx.T@xx/len(xx);cross=xx.T@(y-ym)/len(xx)
    best=None
    # Fixed tie break favors stronger regularization; no evaluation-label access.
    for lam in sorted(lambdas,reverse=True):
        w=np.linalg.solve(cov+lam*np.eye(cov.shape[0]),cross)
        mse=float(np.mean((vv@w+ym-vy)**2))
        if best is None or mse<best[0]:best=(mse,lam,w)
    mse,lam,w=best
    return (tt@w+ym).astype(np.float32),dict(validation_mse=mse,alpha=lam,feature_dim=x.shape[-1],
        head_parameters=int(w.size+ym.size)),(mean,std,w,ym)

# Final implementation source: mte_attribution_resolution_2026_09_12/family_interaction_probe.py:34

def reader(x,y,v):
    mu=x.mean(0);std=x.std(0);std=np.maximum(std,1e-6)
    x=(x-mu)/std;v=(v-mu)/std;ym=y.mean(0)
    w=np.linalg.solve(x.T@x/len(x)+0.01*np.eye(x.shape[1]),x.T@(y-ym)/len(x))
    return v@w+ym

# Final implementation source: mte_attribution_resolution_2026_09_12/interaction_experiment.py:14

def transform(v,M):return np.einsum('cd,bhdek->bhcek',M,v,optimize=True)
