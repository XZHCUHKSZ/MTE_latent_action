"""FLAM-state: numerical tokenizer in front of the unchanged author factored LAM.

Coordinates are feature identities, not spatial pixels or oracle agent slots.
The original tensor axis names H/W are used only to transport a token list.
"""
from copy import deepcopy
import torch
from torch import nn


class CoordinateEncoder(nn.Module):
    def __init__(self, observation_dim, feature_dim=128):
        super().__init__()
        self.H,self.W,self.D=1,observation_dim,feature_dim
        self.value=nn.Sequential(nn.Linear(1,128),nn.GELU(),nn.Linear(128,feature_dim))
        self.coordinate=nn.Parameter(torch.randn(observation_dim,feature_dim)*.02)

    def forward(self, observations):
        if observations.shape[-1]!=self.W:raise ValueError('Numerical coordinate mismatch')
        return (self.value(observations.unsqueeze(-1))+self.coordinate).unsqueeze(-3)


class CoordinateDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,1))

    def forward(self,tokens):return self.readout(tokens).squeeze(-1).squeeze(-2)


class NumericalTokenizer(nn.Module):
    def __init__(self,cfg,observation_dim):
        super().__init__()
        from models.modules.quantizer.fsq import FiniteScalarQuantization
        # Config equality is the author's constructor interface. Actual frontend
        # identity is recorded separately; this is explicitly NOT Impala pixels.
        self.tokenizer_cfg=deepcopy(cfg.tokenizer)
        self.encoder=CoordinateEncoder(observation_dim)
        self.decoder=CoordinateDecoder()
        self.quantizer=FiniteScalarQuantization(cfg.tokenizer.quantizer)
        self.adapter_spec=dict(name='coordinate_MLP_FSQ_state_tokenizer_v1',observation_dim=observation_dim,
                               tokens=observation_dim,token_dim=128,physical_spatial_grid=False,
                               reconstruction='normalized numerical MSE; no image LPIPS objective')

    def forward(self,observations):
        z=self.encoder(observations)
        q,ids,penalty,logging=self.quantizer(z)
        reconstructed=self.decoder(q)
        return (reconstructed-observations).square().mean()+penalty,reconstructed


def factored_model(cfg,tokenizer):
    from models.lam.lam_factored import FactoredLatentActionModel
    model=FactoredLatentActionModel(cfg,tokenizer=tokenizer)
    assert model.lam_cfg.num_slots==4 and model.lam_cfg.d_model==256
    assert model.lam_cfg.num_prediction_steps==5 and model.lam_cfg.image_loss_weight==0
    return model


def training_loss(model,numerical_sequence):
    from data.data_utils.dataclass import SubTrajectory
    # `image` is a legacy author API field; the tensor here is [B,T,D], not RGB.
    model.train()
    return model(SubTrajectory(image=numerical_sequence),is_last_batch=False)[0]


@torch.no_grad()
def transition_code(model,numerical_sequence):
    from data.data_utils.dataclass import SubTrajectory
    model.eval()
    code=model.encode(SubTrajectory(image=numerical_sequence))[3]
    assert code.shape[-2:]==(4,256)
    return code[:,-1,0].flatten(1)
