import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

# Final implementation source: visual_multiagent_2026_09_10/pixel_pipeline_v1/worker.py:20

def teacher(p):
    # Curator/evaluator-only function; no environment imports at module scope.
    from environments.teacher import CentralSACTeacher, sha256_file
    gatepath = ROOT/p['teacher_gate']
    g = json.loads(gatepath.read_text(encoding='utf-8'))
    cp = Path(g['checkpoint'])
    if not cp.is_absolute(): cp = ROOT/cp
    assert g['passed'] and g['scenario']=='Ant' and g['agent_conf']=='4x2'
    assert sha256_file(cp)==g['checkpoint_sha256']
    return CentralSACTeacher(cp,'Ant',device='cpu')

# Final implementation source: visual_multiagent_2026_09_10/pixel_pipeline_v1/worker.py:32

def original_render(env,p,split,seed):
    from .camera import LocalAntVisual
    from .dcs_source import IMAGES
    r = LocalAntVisual(env,seed=seed)
    r.bg._video_paths=[str(IMAGES/s) for s in p['background_videos'][split]]
    r.bg._random_state=np.random.RandomState(seed+1)
    selected=str(np.random.RandomState(seed+1).choice(r.bg._video_paths))
    r.bg._reset_background()
    return r,Path(selected).name

# Final implementation source: visual_multiagent_2026_09_10/seed5_render_repair/render.py:4

def render(*args,samples=0,target_only=False,**kwargs):
    import mujoco
    from OpenGL import GL
    factory=mujoco.Renderer
    def create(model,*a,**kw):
        # Only the separate display model is passed here; native physics untouched.
        model.vis.quality.offsamples=samples
        obj=factory(model,*a,**kw)
        obj._gl_context.make_current();GL.glDisable(GL.GL_DITHER)
        return obj
    mujoco.Renderer=create
    try:r,video=original_render(*args,**kwargs)
    finally:mujoco.Renderer=factory
    r.renderer._gl_context.make_current();GL.glDisable(GL.GL_DITHER)
    assert r.model.vis.quality.offsamples==samples and not GL.glIsEnabled(GL.GL_DITHER)
    if target_only:
        assert r.mounts[0][0]=='agent_0'
        # The deployed policy only reads pixels()[0]. Partners still act in
        # the original joint physics; skip their unused camera renders only.
        r.mounts=r.mounts[:1]
    return r,video
