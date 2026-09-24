"""Explicit OTF-state adaptation using author VQ, factor descriptors and aggregator.

Scalar observation coordinates replace image patches. Coordinate-wise MLPs
replace image convolutions; no renderer, invented RGB, or physical patch grid.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F


class CoordinateTokens(nn.Module):
    def __init__(self,dim,features):
        super().__init__()
        self.grid_height,self.grid_width=1,dim
        self.value=nn.Sequential(nn.Linear(1,128),nn.GELU(),nn.Linear(128,features))
        self.position=nn.Parameter(torch.randn(dim,features)*.02)

    def forward(self,x):
        if x.ndim!=2 or x.shape[1]!=self.grid_width:raise ValueError('Expected[B,numerical_coordinates]')
        return self.value(x.unsqueeze(-1))+self.position


class NumericalMotionReadout(nn.Module):
    def __init__(self,features,hidden):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(features,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,1))

    def forward(self,x):
        # Original factor decoder transport is [B,F,1,D]; no spatial convolution.
        return self.net(x.squeeze(2).transpose(1,2)).squeeze(-1)


class OTFStateVQ(nn.Module):
    def __init__(self,observation_dim,cfg):
        super().__init__()
        from otf_vqvae.model import VectorQuantizer,FactorDecoder
        from omegaconf import OmegaConf
        self.cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
        self.cfg.image_height=1;self.cfg.image_width=observation_dim;self.cfg.channels=1;self.cfg.motion_channels=1
        self.encoder=CoordinateTokens(observation_dim,int(cfg.latent_dim))
        self.reference_encoder=CoordinateTokens(observation_dim,int(cfg.reference_feature_dim))
        self.quantizer=VectorQuantizer(cfg)
        self.decoder=FactorDecoder(self.cfg,1,observation_dim)
        self.decoder.conv_decoder=NumericalMotionReadout(int(cfg.decoder_feature_dim)+int(cfg.reference_feature_dim),int(cfg.decoder_hidden_dim))
        self.num_codes=int(cfg.codebook_size);self.num_patches=observation_dim
        self.summary_eps=float(cfg.summary_eps);self.orth_active_only=bool(cfg.orth_active_only)
        self.register_buffer('patch_coordinates',torch.stack([torch.zeros(observation_dim),torch.arange(observation_dim)],-1).float(),persistent=False)

    @torch.no_grad()
    def initialize(self,current,future,seed):
        from otf_vqvae.model import run_kmeans
        tokens=self.encoder(future-current).flatten(0,1)
        centers=run_kmeans(tokens,self.num_codes,int(self.cfg.kmeans_iters),seed)
        self.quantizer.set_codebook(centers,0)

    def forward(self,current,future,quantized=True):
        from otf_vqvae.model import OTFVQVAE
        motion=future-current;tokens=self.encoder(motion)
        reference=self.reference_encoder(current).transpose(1,2).unsqueeze(2)
        if not quantized:
            prediction=self.decoder.decode_continuous(tokens,reference)
            return dict(loss=(prediction-motion).square().mean(),prediction=prediction,tokens=tokens)
        q=self.quantizer(tokens)
        weights,occupancy,active,embeddings,centroids=OTFVQVAE.summarize_assignments(self,q['indices'],q['quantized_st'])
        prediction=self.decoder.decode_factors(embeddings,weights,occupancy,reference)
        orth=self.quantizer.orthogonality_loss(q['indices'],self.orth_active_only)
        loss=(prediction-motion).square().mean()+q['code_loss']+.25*q['commit_loss']+1e-4*orth
        return dict(loss=loss,prediction=prediction,tokens=tokens,weights=weights,occupancy=occupancy,**q)


class OTFStateLAM(nn.Module):
    def __init__(self,vq,cfg):
        super().__init__()
        from otf_lam_pixels.model import PixelActionAggregator,PixelTokenForwardDecoder
        self.vq=deepcopy(vq).eval()
        for p in self.vq.parameters():p.requires_grad_(False)
        self.state_encoder=CoordinateTokens(vq.num_patches,128)
        self.occupancy_film=nn.Linear(vq.num_codes,256)
        self.action_aggregator=PixelActionAggregator(num_patches=vq.num_patches,state_dim=128,code_dim=int(vq.cfg.latent_dim),cfg=cfg)
        numeric_cfg=dict(cfg);numeric_cfg['decoder_upsample_blocks']=0
        self.forward_decoder=PixelTokenForwardDecoder(num_patches=vq.num_patches,grid_size=(1,vq.num_patches),
            image_size=(1,vq.num_patches),state_feature_dim=128,z_action_dim=256,output_channels=1,cfg=numeric_cfg)
        # Retain original384-wide two-layer conditioned token predictor; only
        # replace the spatial3x3 readout with an independent coordinate MLP.
        self.forward_decoder.output_projection=CoordinateOutputProjection(int(cfg.get('spatial_decoder_hidden_dim',128)))

    def train(self,mode=True):
        super().train(mode);self.vq.eval();return self

    def forward(self,current,future):
        with torch.no_grad():
            tokens=self.vq.encoder(future-current)
            q=self.vq.quantizer(tokens)
            onehot=F.one_hot(q['indices'],self.vq.num_codes).to(current.dtype)
            weights=onehot.mean(1)
        scale,shift=self.occupancy_film(weights).chunk(2,-1)
        state=self.state_encoder(current)*(1+scale[:,None])+shift[:,None]
        z=self.action_aggregator(state,q['quantized'])['z_act']
        prediction=self.forward_decoder(state,z,current[:,None,None,:])[:,0,0]
        return dict(z_act=z,prediction=prediction,loss=(prediction-future).square().mean(),weights=weights,occupancy=onehot.transpose(1,2))


class CoordinateOutputProjection(nn.Module):
    def __init__(self,features):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(features,features),nn.ReLU6(),nn.Linear(features,1))

    def forward(self,x):
        return self.net(x.permute(0,2,3,1)).permute(0,3,1,2)
