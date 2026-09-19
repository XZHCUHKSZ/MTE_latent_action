"""Timing adapters for missing paper baselines; original classes/objectives retained."""
import numpy as np
import torch
from closed_loop_lam_v1 import common as C
from training.temporal_completion_mpe import edge_view
from training.grounding_mpe import train_supervised_history


def train(x,a,b,masks,n,arm,p,out,progress):
    if arm in ('continuous','deepsets','random'):
        native={'continuous':'continuous_lam','deepsets':'matched_cf_deepsets','random':'random_edge_encoder'}[arm]
        view=edge_view(x,a,b,masks,3)
        if arm=='continuous':
            future=x[:,3:25]
            view.update(cf_root=future,cf_ego_null=np.zeros_like(future),cf_other_null=future,cf_both_null=np.zeros_like(future))
        r=C.train_representation(view,native,np.arange(n),p['seed'],z_dim=16,hidden=128,
            updates=p['representation_updates'],batch_size=p['batch'],device='cuda',training_profile='matched',
            requested_policy_updates=p['history_updates'])
        return r.z.reshape(len(x),22,-1),dict(source='unchanged common.train_representation',method=native,
            timing='continuous t to t+2; other methods use h3 matched edges',parameters=r.parameter_count,settings=r.diagnostics)
    assert arm in ('lapo_state','laom_state')
    native=arm+'_adapter';settings=C.resolve_method_training_settings(method=native,profile='matched',
        requested_representation_updates=p['representation_updates'],requested_policy_updates=p['history_updates'],
        batch_size=p['batch'],train_transition_count=n*22,default_lr=1e-3)
    C.seed_all(p['seed']);rng=np.random.default_rng(p['seed']+17)
    mu,sd=C.fit_mean_std(x[:n,1:23].reshape(-1,16));v=torch.tensor((x-mu)/sd,device='cuda')
    net=(C.LAPOStateAdapter(16,128) if arm=='lapo_state' else C.LAOMStateAdapter(16,16,128)).cuda()
    opt=torch.optim.Adam([q for q in net.parameters() if q.requires_grad],lr=settings.representation_lr)
    scheduler=C._profile_scheduler(opt,settings.use_linear_warmup_decay,settings.representation_updates)
    def call(ids):
        ep=ids//22;t=ids%22+1
        if arm=='lapo_state':
            pred,z,_,vq,_,_=net(torch.stack([v[ep,t-1],v[ep,t],v[ep,t+2]],1))
            return z,(pred-v[ep,t+2]).square().mean()+vq
        pred,z,_=net(v[ep,t],v[ep,t+2])
        return z,(pred-net.target(v[ep,t+2]).detach()).square().mean()
    for step in range(settings.representation_updates):
        ids=rng.choice(n*22,min(p['batch'],n*22),replace=n*22<p['batch'])
        _,loss=call(ids);assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
        if scheduler is not None:scheduler.step()
        if arm=='laom_state':net.update_target(settings.laom_target_tau)
        if step%100==0:progress('state_adapter',arm=arm,update=step+1,updates=settings.representation_updates)
    net.eval();blocks=[]
    with torch.no_grad():
        for lo in range(0,len(x)*22,256):blocks.append(call(np.arange(lo,min(lo+256,len(x)*22)))[0].cpu().numpy())
    torch.save(dict(state_dict=net.cpu().state_dict(),mean=mu,std=sd,offset=2,frozen=True),out/'representation.pt')
    return np.concatenate(blocks).reshape(len(x),22,-1),dict(source_class=type(net).__name__,
        settings=settings.to_dict(),timing='Explicit t+2 current-action observability repair; original state-adapter capacity, objective, optimizer, schedule; not official reproduction')


def train_idm_t2(positions,actions,fit,val,seed,updates,policy_updates):
    """Same IDM-relabel learner as grounding_mpe; only future changes to t+2."""
    C.seed_all(seed);mean=positions[:,:23].mean((0,1));std=positions[:,:23].std((0,1)).clip(1e-6)
    x=torch.tensor((positions[:,1:23]-mean)/std,device='cuda')
    nxt=torch.tensor((positions[:,3:25]-mean)/std,device='cuda')
    xx=x[fit].reshape(-1,16);nn=nxt[fit].reshape(-1,16)
    yy=torch.tensor(actions[fit].reshape(-1,2),device='cuda')
    model=C.InverseDynamics(16,2).cuda();opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    rng=np.random.default_rng(seed+919)
    for _ in range(updates):
        ids=rng.integers(0,len(xx),min(512,len(xx)));loss=(model(xx[ids],nn[ids])-yy[ids]).square().mean()
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():pseudo=model(x,nxt).cpu().numpy()
    net,mu,sd,diag=train_supervised_history(positions,pseudo,np.arange(len(positions)),
        positions[val],actions[val],seed+1,policy_updates)
    diag.update(idm_updates=updates,idm_source_class='common.InverseDynamics',inverse_offset=2,
                native_label_permission='fit and validation within B; pseudo labels on observations after grounding begins')
    return net,mu,sd,diag
