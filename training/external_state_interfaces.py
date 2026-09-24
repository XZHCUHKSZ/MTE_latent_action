"""Time and deployment interfaces for explicitly named external state adaptations."""
from dataclasses import dataclass
import numpy as np
import torch
from closed_loop_lam_v1.common import RecurrentPolicy, ActionDecoder


@dataclass(frozen=True)
class StateContract:
    environment: str
    observation_dim: int
    transition_stride: int
    latent_dim: int

    @classmethod
    def create(cls, environment, method):
        if environment not in ('mpe','mamujoco') or method not in ('flam','otf'):
            raise ValueError('Unknown state adaptation')
        return cls(environment,16 if environment=='mpe' else 105,
                   2 if environment=='mpe' else 1,1024 if method=='flam' else 256)

    def window_index(self, bank, length=10):
        span=1+(length-1)*self.transition_stride
        return bank.window_index(span)

    def take(self, bank, index, mean, scale, length=10):
        span=1+(length-1)*self.transition_stride
        return bank.take(index,span,mean,scale)[:,::self.transition_stride].copy()

    def control_times(self, bank):
        if bank.observations.shape[-1]!=self.observation_dim:
            raise ValueError('Wrong environment observation dimension')
        if self.environment=='mpe':
            if bank.observations.shape[1]!=26:raise ValueError('Original MPE requires26 frames')
            return np.arange(1,23,dtype=np.int64)
        return np.arange(bank.observations.shape[1]-1,dtype=np.int64)

    def export_frames(self, time, max_context=10):
        # Last pair is exactly (t,t+stride); earlier frames stay on this stride.
        end=int(time)+self.transition_stride
        start=max(end-(max_context-1)*self.transition_stride,end%self.transition_stride)
        frames=np.arange(start,end+1,self.transition_stride,dtype=np.int64)
        if len(frames)<2 or frames[-2]!=time:raise ValueError('Bad transition time')
        return frames

    def valid_target_mask(self, bank):
        times=self.control_times(bank)
        return np.stack([bank.valid_transitions[:,t:t+self.transition_stride].all(1) for t in times],1)

    def history_observations(self, bank):
        # MPE preserves x0 warmup and trains targets only at x1..x22.
        return bank.observations[:,:23] if self.environment=='mpe' else bank.observations[:,:-1]

    def select_history_outputs(self, values):
        return values[:,1:23] if self.environment=='mpe' else values

    def align_native_actions(self, actions, bank):
        """Curator/downstream only: complete native actions, never pretraining input."""
        if actions.shape!=(len(bank.observations),bank.observations.shape[1]-1,2):
            raise ValueError('Need complete target-only native actions; pre-sliced arrays are rejected')
        return actions[:,self.control_times(bank)].copy()


class StateHistoryController(torch.nn.Module):
    """Existing shared GRU and decoder, streaming one current observation at a time."""
    def __init__(self, contract, mean, scale, latent_mean=None, latent_scale=None):
        super().__init__()
        self.contract=contract
        self.history=RecurrentPolicy(contract.observation_dim,contract.latent_dim,hidden=128)
        self.decoder=ActionDecoder(contract.latent_dim,2,hidden=128)
        self.register_buffer('obs_mean',torch.as_tensor(mean,dtype=torch.float32).clone())
        self.register_buffer('obs_scale',torch.as_tensor(scale,dtype=torch.float32).clone())
        self.register_buffer('latent_mean',torch.zeros(contract.latent_dim) if latent_mean is None else torch.as_tensor(latent_mean,dtype=torch.float32).clone())
        self.register_buffer('latent_scale',torch.ones(contract.latent_dim) if latent_scale is None else torch.as_tensor(latent_scale,dtype=torch.float32).clone())
        if self.obs_mean.shape!=(contract.observation_dim,) or self.obs_scale.shape!=self.obs_mean.shape:
            raise ValueError('Observation normalization shape mismatch')
        if not torch.isfinite(self.obs_mean).all() or not torch.isfinite(self.obs_scale).all() or not (self.obs_scale>0).all():
            raise ValueError('Invalid observation normalization')

    def forward(self, observations, hidden=None):
        if observations.shape[-1]!=self.contract.observation_dim:raise ValueError('Wrong observation dimension')
        z,hidden=self.history((observations-self.obs_mean)/self.obs_scale,hidden)
        return self.decoder((z-self.latent_mean)/self.latent_scale),hidden

    @torch.no_grad()
    def act(self, observation, hidden=None):
        device=self.obs_mean.device
        x=torch.as_tensor(observation,dtype=torch.float32,device=device).reshape(1,1,-1)
        action,hidden=self(x,hidden)
        return action[0,0].clamp(-1,1).cpu().numpy(),hidden
