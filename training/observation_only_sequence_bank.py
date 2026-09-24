"""Numerical observation sequences for explicit baseline adaptations.

Coordinate identities remain numerical features. This module supplies no
images, actor labels, latent-action model, or downstream action alignment.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class ObservationSequenceBank:
    observations: np.ndarray
    valid_transitions: np.ndarray

    @classmethod
    def load(cls, path: Path, observation_key: str):
        assert observation_key in {'positions', 'observations'}
        with np.load(path, allow_pickle=False) as archive:
            allowed={observation_key} | ({'valid_mask'} if observation_key=='observations' else set())
            if set(archive.files)!=allowed:
                raise ValueError('Only the explicitly allowed observation/mask archive is accepted')
            x=archive[observation_key].copy()
            mask=archive['valid_mask'].copy() if 'valid_mask' in archive.files else np.ones(x.shape[:2],bool)[:,:-1]
        if x.ndim!=3 or x.dtype!=np.float32 or mask.dtype!=np.bool_ or mask.shape!=(x.shape[0],x.shape[1]-1):
            raise ValueError('Invalid numerical observation interface')
        bank=cls(x,mask)
        if not np.isfinite(x[bank.valid_frames()]).all():
            raise ValueError('Nonfinite valid observations')
        return bank

    def valid_frames(self):
        m=np.zeros(self.observations.shape[:2],dtype=bool)
        m[:,:-1]|=self.valid_transitions
        m[:,1:]|=self.valid_transitions
        return m

    def fit_normalization(self):
        """Call on the training bank only; never mix in development frames."""
        values=self.observations[self.valid_frames()].astype(np.float64)
        if not len(values):
            raise ValueError('No valid training frames')
        return values.mean(0),np.maximum(values.std(0,ddof=0),1e-6)

    def window_index(self, length):
        if not isinstance(length,int) or length<2 or length>self.observations.shape[1]:
            raise ValueError('Invalid observation window length')
        windows=np.lib.stride_tricks.sliding_window_view(self.valid_transitions,length-1,axis=1)
        return np.argwhere(windows.all(-1)).astype(np.int64)

    def take(self, index, length, mean, scale):
        index=np.asarray(index,dtype=np.int64)
        if index.ndim!=2 or index.shape[1]!=2:
            raise ValueError('Expected episode/start identities')
        mean=np.asarray(mean);scale=np.asarray(scale)
        if mean.shape!=self.observations.shape[2:] or scale.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or not (scale>0).all():
            raise ValueError('Invalid frozen normalization')
        samples=[]
        for episode,start in index:
            if episode<0 or episode>=len(self.observations) or start<0 or start+length>self.observations.shape[1] or length<2:
                raise ValueError('Out-of-range sequence')
            if not self.valid_transitions[episode,start:start+length-1].all():
                raise ValueError('Sequence crosses an invalid transition')
            samples.append((self.observations[episode,start:start+length].astype(np.float64)-mean)/scale)
        if not samples:
            raise ValueError('Empty sequence batch')
        return np.stack(samples).astype(np.float32)
