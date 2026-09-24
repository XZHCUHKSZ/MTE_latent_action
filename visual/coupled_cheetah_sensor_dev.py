"""Development-only RGB adapter; native CoupledHalfCheetah dynamics untouched.

Uses the two cameras from the official XML, not learned entity slots.
DCS source bodies are reused unchanged through the existing compatibility loader.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from OpenGL import GL
from visual.scene import RenderHost
from visual.dcs_source import load


class CameraDataProxy:
    """Present the selected body's COM at DCS's single-body index 1.

    Only the returned COM array is copied. Camera pose writes still reach the
    separate display data; native simulation data is never exposed here.
    """
    def __init__(self, data, body_id):
        self.data, self.body_id = data, int(body_id)

    @property
    def subtree_com(self):
        value = self.data.subtree_com.copy()
        value[1] = value[self.body_id]
        return value

    def __getattr__(self, name):
        return getattr(self.data, name)


class CoupledCheetahSensor:
    def __init__(self, env, *, mode, seed, images=None, split='train'):
        if mode not in ('clean', 'background', 'camera', 'color', 'combined'):
            raise ValueError(mode)
        self.env, self.native = env, env.single_agent_env.unwrapped
        if len(env.possible_agents) != 2 or self.native.model.nu != 12:
            raise ValueError('Requires native CoupledHalfCheetah 1p1')
        self.mode, self.steps = mode, 0
        self.native_materials = self.native.model.mat_rgba.copy()
        self.native_geoms = self.native.model.geom_rgba.copy()
        tree = ET.parse(self.native.fullpath)
        root, asset = tree.getroot(), tree.getroot().find('asset')
        sky = asset.find("texture[@type='skybox']")
        sky.set('height', '800'); sky.set('width', '800')
        # Material compatibility is render-only, preserving native initial RGB.
        for element in root.findall('.//worldbody//geom'):
            name = element.get('name')
            if not name:
                continue
            gid = self.native.model.geom(name).id
            rgba = ' '.join(map(str, self.native.model.geom_rgba[gid]))
            if name == 'floor':
                asset.find("material[@name='MatPlane']").set('rgba', rgba)
            else:
                material = 'sensor_' + name
                ET.SubElement(asset, 'material', name=material, rgba=rgba)
                element.set('material', material)
            element.set('rgba', '0.5 0.5 0.5 1')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        assert (self.model.nq, self.model.nv, self.model.nu) == (18, 18, 12)
        self.camera_ids = [self.model.camera(name).id for name in ('track0', 'track1')]
        self.body_ids = [int(self.model.cam_bodyid[cid]) for cid in self.camera_ids]
        assert self.body_ids == [self.model.body('torso0').id, self.model.body('torso1').id]
        assert all(self.model.cam_mode[cid] == mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM for cid in self.camera_ids)
        self.model.vis.quality.offsamples = 0
        self.renderer = mujoco.Renderer(self.model, height=64, width=64)
        self.renderer._gl_context.make_current(); GL.glDisable(GL.GL_DITHER)
        self.sync()
        self.hosts = [RenderHost(self.model, self.data, self.renderer) for _ in range(2)]
        for host, bid in zip(self.hosts, self.body_ids):
            host.physics.named.data = CameraDataProxy(self.data, bid)
        self.bg, self.color, self.cameras = None, None, []
        primary = self.hosts[0]
        if mode in ('background', 'combined'):
            if images is None or not Path(images).is_dir():
                raise ValueError('Real DAVIS assets required; no synthetic fallback')
            self.bg = load('background', mujoco).DistractingBackgroundEnv(
                primary, dataset_path=str(images), dataset_videos=split,
                num_videos=None, dynamic=True, video_alpha=1.,
                ground_plane_alpha=1., seed=seed+1)
            primary = self.bg
        utils = load('suite_utils')
        scale = utils.DIFFICULTY_SCALE['scale_easy_video_hard']
        if mode in ('camera', 'combined'):
            for index, cid in enumerate(self.camera_ids):
                camera = load('camera').DistractingCameraEnv(
                    primary if index == 0 else self.hosts[1], camera_id=cid,
                    seed=seed+2, **utils.get_camera_kwargs('cheetah', scale, True))
                self.cameras.append(camera)
            primary = self.cameras[0]
        if mode in ('color', 'combined'):
            self.color = load('color').DistractingColorEnv(
                primary, seed=seed+3, **utils.get_color_kwargs(scale, True))
            self.color._random_state = np.random.RandomState(seed+3)
            primary = self.color
        self.primary = primary
        primary.reset()
        if self.cameras:
            self.cameras[1].reset()

    def sync(self):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.native.model, spec))
        mujoco.mj_getState(self.native.model, self.native.data, state, spec)
        mujoco.mj_setState(self.model, self.data, state, spec)
        mujoco.mj_forward(self.model, self.data)

    def advance(self):
        self.sync()
        self.primary.step(None)
        if self.cameras:
            self.cameras[1].step(None)
        self.steps += 1

    def pixels(self):
        self.renderer._gl_context.make_current(); GL.glDisable(GL.GL_DITHER)
        frames = []
        for cid in self.camera_ids:
            self.renderer.update_scene(self.data, camera=cid)
            frames.append(self.renderer.render().copy())
        assert np.array_equal(self.native.model.mat_rgba, self.native_materials)
        assert np.array_equal(self.native.model.geom_rgba, self.native_geoms)
        return np.stack(frames)

    def close(self):
        self.renderer.close()
