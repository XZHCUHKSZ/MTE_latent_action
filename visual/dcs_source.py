import ast
import collections
import copy
import glob
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OFFICIAL = Path(__file__).resolve().parents[1] / 'third_party/laom'
ASSETS = HERE / 'assets'
IMAGES = ASSETS / 'DAVIS/JPEGImages/480p'
URL = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip'
ARCHIVE_BYTES = 832766765

# Final implementation source: visual_multiagent_2026_09_10/laom_complexity_v1/source.py:27

def load(name, mujoco=None):
    path = OFFICIAL / f'src/dcs/{name}.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    # Imports alone are replaced; original function/class bodies are unchanged.
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    ns = dict(np=np, copy=copy, collections=collections, glob=glob, os=os,
              Image=Image, control=SimpleNamespace(Environment=object),
              mjbindings=SimpleNamespace(mjlib=mujoco))
    exec(compile(tree, str(path), 'exec'), ns)
    return SimpleNamespace(**ns)

# Final implementation source: visual_multiagent_2026_09_10/laom_complexity_v1/source.py:39

def manifest():
    utils, bg = load('suite_utils'), load('background')
    name = 'scale_easy_video_hard'
    scale = utils.DIFFICULTY_SCALE[name]
    # Supported domains give the same camera values except reacher's quadrant.
    # This is a reference profile, NOT a claim that Ant is officially supported.
    camera = utils.get_camera_kwargs('walker', scale, True)
    assert camera == utils.get_camera_kwargs('humanoid', scale, True)
    return dict(reference_repository='https://github.com/dunnolab/laom',
        archived_source=str(OFFICIAL), difficulty=name, dynamic=True,
        scale=scale, num_videos=utils.DIFFICULTY_NUM_VIDEOS[name],
        train_videos=bg.DAVIS17_TRAINING_VIDEOS,
        validation_videos=bg.DAVIS17_VALIDATION_VIDEOS,
        camera=camera, color=utils.get_color_kwargs(scale, True),
        background=dict(video_alpha=1., shuffle=False, random_start=True,
            random_direction=True, boundary='official ping-pong',
            ground_alpha_by_reference_domain={'walker/cheetah/hopper':1.,
                'reacher':0., 'other supported domains':.3}),
        image_size=[64,64], frame_stack=3, archive_url=URL,
        archive_bytes=ARCHIVE_BYTES,
        ant_status='not an official DCS domain; explicit multi-agent adaptation',
        differences=['Ant physics, reward and 4x2 action factorization retained',
            'render-only material assignment for Ant bodies, absent in native XML',
            'ground alpha 0.3 chosen from non-locomotion-specific DCS fallback; not official Ant value',
            'tex_rgb compatibility alias to native tex_data, original source untouched',
            'color wrapper ignores seed argument upstream; adapter explicitly seeds its RNG',
            'visual protocol matching does not establish equal task difficulty or full LAOM reproduction'],
        learner_contract='RGB/time/fixed-agent identity only; no simulator queries during pretraining',
        training_status='not started; rendering and asset qualification first')

# Final implementation source: visual_multiagent_2026_09_10/laom_complexity_v1/source.py:70

def check_assets():
    m = manifest()
    train, val = set(m['train_videos']), set(m['validation_videos'])
    assert len(train)==60 and len(val)==30 and not train.intersection(val)
    counts = {s:len(list((IMAGES/s).glob('*.jpg'))) for s in sorted(train|val)}
    missing = [s for s,n in counts.items() if n<2]
    return dict(ready=not missing, root=str(IMAGES), video_counts=counts,
                missing=missing, expected_train=60, expected_validation=30)
