from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .dcs_source import IMAGES, load, manifest, check_assets

# Final implementation source: visual_multiagent_2026_09_10/laom_complexity_v1/render.py:14

class ModelProxy:
    def __init__(self, model):self.model=model
    @property
    def tex_rgb(self):return self.model.tex_data
    @property
    def ptr(self):return self.model
    def __getattr__(self,name):return getattr(self.model,name)

# Final implementation source: visual_multiagent_2026_09_10/laom_complexity_v1/render.py:23

class MaterialNames:
    def __init__(self, model):self.model=model
    def __setitem__(self,key,value):
        name,channel=key
        if name=='grid':name='MatPlane'
        self.model.mat_rgba[self.model.material(name).id,'rgba'.index(channel)]=value

# Final implementation source: visual_multiagent_2026_09_10/laom_complexity_v1/render.py:31

class CurrentContext:
    def __init__(self,renderer):self.renderer=renderer
    def __enter__(self):
        self.renderer._gl_context.make_current()
        return SimpleNamespace(call=lambda fn,*a:fn(*a))
    def __exit__(self,*a):return False

# Final implementation source: visual_multiagent_2026_09_10/laom_complexity_v1/render.py:39

class RenderHost:
    def __init__(self,model,data,renderer):
        self.model,self.data=model,data
        self.physics=SimpleNamespace(model=ModelProxy(model),named=SimpleNamespace(
            data=data,model=SimpleNamespace(mat_rgba=MaterialNames(model))),
            contexts=SimpleNamespace(gl=SimpleNamespace(make_current=lambda:CurrentContext(renderer)),
                                     mujoco=SimpleNamespace(ptr=renderer._mjr_context)))
        self._physics=self.physics
    def reset(self):return SimpleNamespace(first=lambda:True)
    def step(self,action):return SimpleNamespace(first=lambda:False)

# Final implementation source: visual_multiagent_2026_09_10/laom_complexity_v1/render.py:51

class AntVisual:
    def __init__(self,env,split='train',seed=0,background=True):
        if split not in ['train','val']:raise ValueError(split)
        if background and not check_assets()['ready']:raise RuntimeError('DAVIS not ready; no synthetic fallback')
        self.env=env;self.native=env.single_agent_env.unwrapped
        if list(env.possible_agents)!=[f'agent_{i}' for i in range(4)]:raise ValueError('Requires native Ant 4x2')
        self.original=bytes(self.native.model.mat_rgba)
        xml=ET.parse(self.native.fullpath);root=xml.getroot();asset=root.find('asset')
        sky=asset.find("texture[@type='skybox']")
        # Allocate the DCS 800-wide skybox BEFORE compiling, rather than writing
        # an 800-high texture into the small native Ant allocation.
        sky.set('height','800');sky.set('width','800')
        for element in root.findall('.//worldbody//geom'):
            name=element.get('name')
            if not name:continue
            gid=self.native.model.geom(name).id
            rgba=' '.join(map(str,self.native.model.geom_rgba[gid]))
            if name=='floor':
                asset.find("material[@name='MatPlane']").set('rgba',rgba)
                element.set('rgba','0.5 0.5 0.5 1')
            else:
                mat='visual_'+name
                ET.SubElement(asset,'material',name=mat,rgba=rgba)
                element.set('material',mat);element.set('rgba','0.5 0.5 0.5 1')
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        self.data=mujoco.MjData(self.model)
        assert (self.model.nq,self.model.nv,self.model.nu)==(self.native.model.nq,self.native.model.nv,self.native.model.nu)
        self.renderer=mujoco.Renderer(self.model,height=64,width=64)
        self.host=RenderHost(self.model,self.data,self.renderer)
        m=manifest();self.sync()
        base=self.host;self.bg=None
        if background:
            self.bg=load('background',mujoco).DistractingBackgroundEnv(base,
                dataset_path=str(IMAGES),dataset_videos=split,num_videos=None,
                dynamic=True,video_alpha=1.,ground_plane_alpha=.3,seed=seed+1)
            base=self.bg
        self.camera=load('camera').DistractingCameraEnv(base,camera_id=0,seed=seed+2,**m['camera'])
        self.color=load('color').DistractingColorEnv(self.camera,seed=seed+3,**m['color'])
        # Upstream accepts but ignores seed; explicit, documented reproducibility fix.
        self.color._random_state=np.random.RandomState(seed+3)
        self.color.reset()
        self.steps=0

    def sync(self):
        # Curator render operation only; these arrays are never learner features.
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        state=np.empty(mujoco.mj_stateSize(self.native.model,spec))
        mujoco.mj_getState(self.native.model,self.native.data,state,spec)
        mujoco.mj_setState(self.model,self.data,state,spec)
        mujoco.mj_forward(self.model,self.data)

    def advance(self):
        """Call exactly once AFTER each native parallel step."""
        self.sync();self.color.step(None);self.steps+=1

    def pixels(self):
        self.renderer.update_scene(self.data,camera=0)
        result=self.renderer.render().copy()
        assert result.shape==(64,64,3) and result.dtype==np.uint8
        assert bytes(self.native.model.mat_rgba)==self.original
        return result

    def close(self):self.renderer.close()
