"""Agent-count interfaces; unchanged accepted BC/history/decoder scientific classes."""
import numpy as np
import torch
from closed_loop_lam_v1.common import RecurrentPolicy,seed_all

def train_supervised_history(positions, targets, fit, validation_positions, validation_actions, seed, updates=1200):
    """Original GRU class and BC MSE, with common x0 warmup and counted validation."""
    seed_all(seed)
    obs=positions[:,:23]
    mean=obs.mean((0,1));std=obs.std((0,1)).clip(1e-6)
    x=torch.as_tensor((obs-mean)/std,device='cuda');y=torch.as_tensor(targets,device='cuda')
    vx=torch.as_tensor((validation_positions[:,:23]-mean)/std,device='cuda')
    vy=torch.as_tensor(validation_actions,device='cuda')
    net=RecurrentPolicy(positions.shape[-1],2).cuda();opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    rng=np.random.default_rng(seed+313);best=float('inf');selected=0;best_state=None
    for step in range(1,updates+1):
        ix=rng.choice(fit,min(16,len(fit)),replace=False)
        pred,_=net(x[ix]);loss=(pred[:,1:]-y[ix]).square().mean()
        if not torch.isfinite(loss): raise ValueError('BC nonfinite')
        opt.zero_grad();loss.backward();opt.step()
        if step==1 or step%25==0 or step==updates:
            with torch.no_grad():
                pred,_=net(vx);candidate=(pred[:,1:]-vy).square().mean().item()
            if candidate<best:
                best=candidate;selected=step;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net.cpu().eval(),mean,std,dict(validation_action_mse=best,selected_update=selected,updates=updates,
        source_class='closed_loop_lam_v1.common.RecurrentPolicy',objective='BC MSE on t=1..22; x0 history warmup')

class ScaledGroundedPolicy:
    def __init__(self,net,mean,std,decoder=None):
        self.net=net.cpu().eval();self.mean=np.asarray(mean);self.std=np.asarray(std);self.decoder=decoder
    def act(self,position,hidden):
        with torch.no_grad():
            x=torch.as_tensor((position-self.mean)/self.std,dtype=torch.float32).reshape(1,1,len(self.mean))
            pred,hidden=self.net(x,hidden)
            action=pred[:,-1] if self.decoder is None else self.decoder(pred[:,-1])
        return action.numpy().reshape(2).clip(-1,1).astype(np.float32),hidden
