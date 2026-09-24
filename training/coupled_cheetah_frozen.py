"""Two-camera native CoupledHalfCheetah adapter; original scientific classes unchanged.

Frozen-only publication API. Both cameras are causal observations for every method. Learned visual endpoints
are predictions under latent replacements, never physical interventions.
"""
from mte.method_names import resolve_coupled_arm, resolve_coupled_ground
from pathlib import Path
import copy,time,hashlib
import numpy as np
import torch
from torch.nn import functional as F
from closed_loop_lam_v1 import common as C
from training.composition import FrozenPair,restore
from training.representation_mamujoco import train_history_policy
from mte.frontends import ObservationReadout
from visual.official import load,load_augmenter
from visual.history import history,OnlineHistory
from utils.access import guard
from utils.atomic import atomic_json as save
METHODS=['edge_cara','edge_cara_mobius_simple','edge_cara_mobius_graph','edge_cara_mobius_tree','edge_cara_mif','continuous_lam','lapo_state_adapter','laom_state_adapter']
OBS={'continuous_lam','lapo_state_adapter','laom_state_adapter'}

def digest_state(net):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in net.state_dict().values())).hexdigest()

def make_frontend():
    return load()['LAOM']((9,64,64),latent_act_dim=32,encoder_channels=(16,32,64),encoder_num_res_blocks=1,act_head_dim=128,obs_head_dim=128)

def pool(net,x):return F.adaptive_avg_pool2d(net.encoder(x).reshape(len(x),64,8,8),(2,2)).flatten(1)

def export_labels(data,out,budget):
    # Privileged preparation boundary; a learner may open only the exported budget.
    y=np.load(data/'target_actions.npy');assert y.shape==(68,200,6)
    out.mkdir(parents=True,exist_ok=False);np.save(out/'target_actions.npy',y[:budget].copy())
    return dict(target_labels=budget*200,partner_labels=0,selection='first fixed fit episodes; no return selection')

def pretrain(c,seed,data,out,note):
    C.seed_all(seed);torch.set_num_threads(c['threads']);dev=c['device'];n=64;steps=200;views=2;d=64
    net=make_frontend().to(dev);target=copy.deepcopy(net).eval()
    for q in target.parameters():q.requires_grad_(False)
    aug=load_augmenter();opt=torch.optim.Adam(net.parameters(),lr=1e-4)
    audit=guard(out,[data/'rgb.npy'],True);rgb=np.load(data/'rgb.npy',mmap_mode='r');assert rgb.shape==(68,201,2,64,64,3)
    rng=np.random.default_rng(seed)
    def tensor(xs):return torch.from_numpy(np.stack(xs)).to(dev).float()/127.5-1
    for u in range(1,c['frontend_updates']+1):
        ids=rng.integers(n*steps*views,size=32);e=ids//400;t=ids//2%200;j=ids%2;k=np.array([rng.integers(1,min(10,200-tt)+1) for tt in t])
        x=tensor([history(rgb[ee],tt,jj) for ee,tt,jj in zip(e,t,j)]);nx=tensor([history(rgb[ee],tt+1,jj) for ee,tt,jj in zip(e,t,j)]);fu=tensor([history(rgb[ee],tt+kk,jj) for ee,tt,jj,kk in zip(e,t,j,k)])
        pred,_,_=net(aug(x),aug(fu))
        with torch.no_grad():yt=target.encoder(aug(nx))
        loss=F.mse_loss(pred,yt);assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
        with torch.no_grad():
            for a,b in zip(target.parameters(),net.parameters()):a.lerp_(b,.001)
        if u==1 or u%100==0 or u==c['frontend_updates']:note(stage='frontend',update=u,updates=c['frontend_updates'])
    net.eval();torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},training_seed=seed,native_action_labels_read=0,simulator_queries=0,updates=c['frontend_updates']),out/'frontend.pt')
    ff=np.empty((68,201,2,256),np.float32);zz=np.empty((68,200,2,32),np.float32)
    with torch.no_grad():
        for e in range(68):
            for j in range(2):
                xs=tensor([history(rgb[e],t,j) for t in range(201)])
                for lo in range(0,201,32):
                    hi=min(lo+32,201);ff[e,lo:hi,j]=pool(net,xs[lo:hi]).cpu().numpy();stop=min(hi,200)
                    if lo<stop:zz[e,lo:stop,j]=net.label(xs[lo:stop],xs[lo+1:stop+1]).cpu().numpy()
            if e%8==0:note(stage='features',episode=e+1,episodes=68)
        f=torch.from_numpy(ff).to(dev);train=f[:n].reshape(-1,256);mean=train.mean(0);cov=(train-mean).T@(train-mean)/(len(train)-1);eig,vec=torch.linalg.eigh(cov);basis=vec[:,-32:];scale=eig[-32:].clamp_min(1e-8).sqrt();x=((f-mean)@basis/scale).cpu().numpy().astype(np.float32)
    torch.save(dict(mean=mean.cpu(),basis=basis.cpu(),scale=scale.cpu(),native_action_labels_read=0),out/'projection.pt');np.savez_compressed(out/'features.npz',x=x,z=zz)
    flat=x.reshape(68,201,64);z=zz.reshape(68,200,64);h=np.concatenate([flat[:,np.maximum(np.arange(200)-1,0)],flat[:,:200]],-1)
    def norm(a):
        m=a[:n].reshape(-1,a.shape[-1]).mean(0);s=a[:n].reshape(-1,a.shape[-1]).std(0).clip(.0001);return torch.as_tensor((a-m)/s,device=dev),m,s
    ht,hm,hs=norm(h);zt,zm,zs=norm(z);bank=rng.choice(n*200,1024,replace=False);bank_ep=bank//200;hh=ht.reshape(-1,128);az=zt.reshape(-1,2,32)
    donors=[]
    with torch.no_grad():
        for lo in range(0,13600,128):
            hi=min(lo+128,13600);dist=torch.cdist(hh[lo:hi],hh[bank]);dist.masked_fill_(torch.as_tensor(np.arange(lo,hi)[:,None]//200==bank_ep[None],device=dev),float('inf'));donors.extend(bank[dist.argmin(1).cpu().numpy()].tolist())
    donors=np.array(donors);assert (donors<12800).all() and (donors//200!=np.arange(13600)//200).all()
    masks=np.array([[0,0],[0,1]],bool);aa=np.empty((68,200,3,2,2,32),np.float32);bb=np.empty_like(aa);readouts=[]
    for horizon in (1,2,3):
        C.seed_all(seed+horizon);valid_t=201-horizon;y=flat[:,horizon:]-flat[:,:valid_t];yt,ym,ys=norm(y);rd=ObservationReadout(64,64,64).to(dev);op=torch.optim.Adam(rd.parameters(),lr=.001);rr=np.random.default_rng(seed+horizon)
        th=ht[:n,:valid_t].reshape(-1,128);tz=zt[:n,:valid_t].reshape(-1,64);ty=yt[:n].reshape(-1,64)
        for u in range(1,c['readout_updates']+1):
            ids=torch.as_tensor(rr.integers(len(th),size=256),device=dev);loss=F.mse_loss(rd(th[ids],tz[ids]),ty[ids]);assert torch.isfinite(loss);op.zero_grad();loss.backward();op.step()
        rd.eval()
        with torch.no_grad():
            dr=az[torch.as_tensor(donors,device=dev)]
            for lo in range(0,13600,128):
                hi=min(lo+128,13600)
                for ci,mask in enumerate(masks):
                    va=az[lo:hi].clone();va[:,mask]=dr[lo:hi,mask];vb=va.clone();vb[:,0]=dr[lo:hi,0];assert torch.equal(va[:,1:],vb[:,1:])
                    aa.reshape(13600,3,2,2,32)[lo:hi,horizon-1,ci]=rd(hh[lo:hi],va.flatten(1)).cpu().numpy().reshape(-1,2,32)
                    bb.reshape(13600,3,2,2,32)[lo:hi,horizon-1,ci]=rd(hh[lo:hi],vb.flatten(1)).cpu().numpy().reshape(-1,2,32)
            dv=float(F.mse_loss(rd(ht[n:,:valid_t].reshape(-1,128),zt[n:,:valid_t].reshape(-1,64)),yt[n:].reshape(-1,64)))
        torch.save(dict(state_dict=rd.cpu().state_dict(),history_mean=hm,history_std=hs,code_mean=zm,code_std=zs,delta_mean=ym,delta_std=ys,horizon=horizon),out/f'readout_h{horizon}.pt');readouts.append(dict(horizon=horizon,updates=c['readout_updates'],dev_mse=dv));note(stage='matched_readout',horizon=horizon)
    assert np.isfinite(aa).all() and np.isfinite(bb).all()
    # The reshape above must address the strided view in place; verify meaningful edges.
    assert float(np.abs(aa-bb).mean())>1e-10,'Collapsed learned endpoint contrast'
    np.savez_compressed(out/'endpoints.npz',actual=aa,reference=bb,masks=masks,donor_indices=donors.reshape(68,200))
    a=aa[:,:,0].reshape(68,200,2,64);b=bb[:,:,0].reshape(68,200,2,64);valid=np.ones((68,200),bool);rows=[]
    for method in ['base']+METHODS:
        C.seed_all(seed)
        if method=='base':lat=z;diag={'source':'two original visual LAOM view latents concatenated','latent_dim':64}
        else:
            if method in OBS:
                nxt=flat[:,1:];data_view=dict(cf_root=nxt,cf_ego_null=np.zeros_like(nxt),cf_other_null=nxt,cf_both_null=np.zeros_like(nxt))
            else:data_view=dict(cf_root=a[:,:,0],cf_ego_null=b[:,:,0],cf_other_null=a[:,:,-1],cf_both_null=b[:,:,-1],cf_context_root=a,cf_context_ego_null=b,cf_context_masks=masks)
            data_view.update(obs=flat,actions=np.zeros((68,200,6),np.float32),valid_step_mask=valid)
            if method=='edge_cara_mif':data_view.update(cf_horizon_entity_root=aa,cf_horizon_entity_ego_null=bb,mif_entity_adjacency=np.ones((2,2),np.float32),effect_horizons=np.array([1,2,3]))
            result=C.train_representation(data_view,method,np.arange(n),seed,z_dim=16,hidden=128,updates=c['representation_updates'],batch_size=128,device=dev,training_profile='matched',requested_policy_updates=c['history_updates']);lat=result.z.reshape(68,200,-1);diag=dict(result.diagnostics,parameters=result.parameter_count)
        assert np.isfinite(lat).all();row=train_history_policy(flat,lat,valid,n,seed,c['history_updates'],out/method/'policy',lambda st,**kw:note(stage=st,method=method,**kw),device=dev,method='edge_cara' if method=='base' else method);rows.append(dict(method=method,representation=diag,policy=row));note(stage='pretrained_method',method=method)
    assert not audit['violations'];save(out/'access_audit.json',audit)
    return dict(methods=rows,readouts=readouts,native_action_labels_read=0,partner_action_labels_read=0,simulator_queries=0,frontend_updates=c['frontend_updates'],representation_updates=c['representation_updates'],history_updates=c['history_updates'],qualification=c['qualification'],data_shape=list(rgb.shape),endpoint_evidence='predicted visual increments under latent replacements; not physical counterfactuals',deployment='same two camera features and causal history for all methods')

def compositions():return ['base']+[q+'_'+m for m in METHODS for q in ['solo','aux']]
def arms():return [a+'__'+route for a in compositions() for route in ['Frozen']]+['bc','idm']
def pair(pre,arm,dev):
    arm = resolve_coupled_arm(arm)
    if arm=='base':net,ck=restore(pre/'base/policy/policy.pt');return FrozenPair(net).to(dev),ck
    route,method=arm.split('_',1);net,ck=restore(pre/method/'policy/policy.pt')
    if route=='solo':return FrozenPair(net).to(dev),ck
    base,bck=restore(pre/'base/policy/policy.pt');assert np.array_equal(ck['obs_mean'],bck['obs_mean']) and np.array_equal(ck['obs_std'],bck['obs_std']);return FrozenPair(base,net).to(dev),ck

def ground(c,seed,pre,labelpath,out,arm,note):
    arm = resolve_coupled_ground(arm)
    C.seed_all(seed);torch.set_num_threads(c['threads']);dev=c['device'];b=c['budget_episodes'];files=[pre/'features.npz',labelpath]+list(pre.glob('*/policy/policy.pt'))
    audit=guard(out,files,False);audit.update(phase='target_only_grounding',partner_labels_read=0,simulator_queries=0)
    x=np.load(pre/'features.npz')['x'].reshape(68,201,64);y=torch.as_tensor(np.load(labelpath),device=dev);assert y.shape==(b,200,6);net=None;curve=[];rng=np.random.default_rng(seed);stream=hashlib.sha256()
    if arm in ['bc','idm']:
        om=x[:64,:200].reshape(-1,64).mean(0);os=x[:64,:200].reshape(-1,64).std(0).clip(.0001);inp=torch.as_tensor((x[:,:200]-om)/os,device=dev);model=C.RecurrentPolicy(64,6).to(dev);mean=scale=None
        if arm=='idm':
            inverse=C.InverseDynamics(64,6).to(dev);op=torch.optim.Adam(inverse.parameters(),lr=.001);xx=torch.as_tensor((x-om)/os,device=dev)
            for u in range(c['ground_updates']):
                ix=torch.as_tensor(rng.integers(b*200,size=256),device=dev);loss=F.mse_loss(inverse(xx[:b,:200].reshape(-1,64)[ix],xx[:b,1:].reshape(-1,64)[ix]),y.reshape(-1,6)[ix]);op.zero_grad();loss.backward();op.step()
            with torch.no_grad():pseudo=inverse(xx[:64,:200],xx[:64,1:]).detach()
            torch.save(dict(state_dict=inverse.cpu().state_dict()),out/'idm.pt');C.seed_all(seed);model=C.RecurrentPolicy(64,6).to(dev)
        initial=None;route=None
    else:
        comp,route=arm.split('__');assert route=='Frozen';net,ck=pair(pre,comp,dev);om,os=ck['obs_mean'],ck['obs_std'];inp=torch.as_tensor((x[:b,:200]-om)/os,device=dev)
        with torch.no_grad():values=net(inp)[0].reshape(b*200,-1)
        mean=values.mean(0);scale=values.std(0).clamp_min(.0001);initial=digest_state(net)
        for q in net.parameters():q.requires_grad_(False)
        net.train(False)
        C.seed_all(seed);model=C.ActionDecoder(values.shape[-1],6,hidden=128).to(dev)
    opt=torch.optim.Adam(list(model.parameters()),lr=.001)
    for u in range(1,c['ground_updates']+1):
        ix=torch.as_tensor(rng.integers(b*200,size=256),device=dev);stream.update(ix.cpu().numpy().tobytes())
        if arm=='idm':
            ids=rng.integers(64,size=8);loss=F.mse_loss(model(inp[ids])[0],pseudo[ids])
        elif arm=='bc':loss=F.mse_loss(model(inp[:b])[0].reshape(-1,6)[ix],y.reshape(-1,6)[ix])
        else:
            v=values;loss=F.mse_loss(model(((v-mean)/scale)[ix]),y.reshape(-1,6)[ix])
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
        if u==1 or u%100==0 or u==c['ground_updates']:curve.append(dict(update=u,loss=float(loss.detach())));note(stage='ground',arm=arm,update=u,updates=c['ground_updates'])
    model.eval();changed=None
    if net is not None:
        net.eval();changed=digest_state(net)!=initial;assert changed==(False)
        with torch.no_grad(), torch.backends.cudnn.flags(allow_tf32=False):
            seq=net(inp[:1])[0];h=None;parts=[]
            for t in range(200):a,h=net(inp[:1,t:t+1],h);parts.append(a)
            error=float((seq-torch.cat(parts,1)).abs().max());assert error<1e-4
    else:error=None
    torch.save(dict(state_dict=model.cpu().state_dict(),history_state=None if net is None else net.cpu().state_dict(),arm=arm,obs_mean=om,obs_std=os,mean=None if mean is None else mean.cpu(),scale=None if scale is None else scale.cpu(),latent_dim=None if net is None else len(mean),updates=c['ground_updates'],qualification=c['qualification'],selection='fixed final update'),out/'controller.pt');save(out/'curve.json',curve);assert not audit['violations'];save(out/'access_audit.json',audit)
    return dict(arm=arm,updates=c['ground_updates'],target_labels=b*200,partner_labels=0,simulator_queries=0,history_changed=changed,history_sequence_error=error,initial_history_sha256=initial,minibatch_sha256=stream.hexdigest(),qualification=c['qualification'],idm_extra_policy_updates=c['ground_updates'] if arm=='idm' else 0)

class Controller:
    def __init__(self,pre,path):
        self.pre=Path(pre);ck=torch.load(path,map_location='cpu',weights_only=False);self.ck=ck;self.arm=ck['arm'];self.hist=None;self.frames=OnlineHistory();self.front=make_frontend();self.front.load_state_dict(torch.load(self.pre/'frontend.pt',map_location='cpu',weights_only=False)['state_dict']);self.front.eval();self.pr=torch.load(self.pre/'projection.pt',map_location='cpu',weights_only=False)
        if self.arm in ['bc','idm']:self.model=C.RecurrentPolicy(64,6);self.net=None
        else:self.net,_=pair(self.pre,self.arm.split('__')[0],'cpu');self.net.load_state_dict(ck['history_state']);self.net.eval();self.model=C.ActionDecoder(ck['latent_dim'],6,hidden=128)
        self.model.load_state_dict(ck['state_dict']);self.model.eval()
    def reset(self):self.frames.reset();self.hist=None
    @torch.no_grad()
    def feature(self,rgb):
        frames=self.frames.observe(rgb);f=pool(self.front,torch.from_numpy(frames).float()/127.5-1);p=self.pr;return ((f-p['mean'])@p['basis']/p['scale']).reshape(1,1,64)
    @torch.no_grad()
    def act_feature(self,x):
        ck=self.ck;xx=(x-torch.as_tensor(ck['obs_mean']))/torch.as_tensor(ck['obs_std'])
        if self.net is None:a,self.hist=self.model(xx,self.hist)
        else:v,self.hist=self.net(xx,self.hist);a=self.model((v-ck['mean'])/ck['scale'])
        return np.clip(a.reshape(-1).numpy(),-1,1)
    def act(self,rgb):return self.act_feature(self.feature(rgb))


def public_arms():
    """CLI-ready aliases for the existing Frozen-only arm list."""
    from mte.method_names import paper_name
    return [paper_name(a.split("__")[0]) + "__Frozen" if "__" in a else a for a in arms()]
