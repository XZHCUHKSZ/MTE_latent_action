from collections import deque
import numpy as np

# Final implementation source: visual_multiagent_2026_09_10/pixel_pipeline_v1/data.py:6

def validate_episode(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 5 or rgb.shape[2:] != (64, 64, 3):
        raise ValueError('Require uint8 [T+1,views,64,64,3] RGB; no numeric state features')
    if rgb.shape[0] < 2 or rgb.shape[1] < 1:
        raise ValueError('Empty views or no complete transition')
    return rgb

# Final implementation source: visual_multiagent_2026_09_10/pixel_pipeline_v1/data.py:14

def history(rgb, t, view):
    """Three frames ending at t; no future reads and no cross-episode padding."""
    validate_episode(rgb)
    if not 0 <= t < len(rgb) or not 0 <= view < rgb.shape[1]:
        raise IndexError((t, view))
    frames = [rgb[max(0, t-k), view] for k in (2, 1, 0)]
    return np.concatenate(frames, axis=-1).transpose(2, 0, 1).copy()

# Final implementation source: visual_multiagent_2026_09_10/pixel_pipeline_v1/data.py:23

def inverse_example(rgb, t, view, offset):
    if not isinstance(offset, (int, np.integer)) or offset < 1 or t + offset >= len(rgb):
        raise ValueError('Future must remain inside this episode')
    # The forward target stays t+1, including when the inverse future is t+k.
    return history(rgb, t, view), history(rgb, t+offset, view), history(rgb, t+1, view)

# Final implementation source: visual_multiagent_2026_09_10/pixel_pipeline_v1/data.py:30

class OnlineHistory:
    """Consumes one observed frame at a time. Reset explicitly at episode start."""
    def __init__(self):
        self.frames = deque(maxlen=3)
        self.shape = None

    def reset(self):
        self.frames.clear()
        self.shape = None

    def observe(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[1:] != (64, 64, 3):
            raise ValueError('Require uint8 [views,64,64,3]')
        if self.shape is not None and rgb.shape != self.shape:
            raise ValueError('View count changed inside episode')
        self.shape = rgb.shape
        if not self.frames:
            self.frames.extend([rgb.copy(), rgb.copy()])
        self.frames.append(rgb.copy())
        return np.concatenate(list(self.frames), axis=-1).transpose(0, 3, 1, 2).copy()
