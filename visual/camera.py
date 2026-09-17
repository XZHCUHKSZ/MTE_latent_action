import numpy as np
from .scene import AntVisual

# Final implementation source: visual_multiagent_2026_09_10/pixel_pipeline_v1/local_render.py:6

class LocalAntVisual(AntVisual):
    """Four fixed-mount views, tied to public native Ant actuator partitions.

    Mounts follow torso pose, not joints or target outcomes. DCS pose jitter is
    transformed into each rig frame; video/color are shared physical appearance.
    This is a documented local-camera adaptation, not an official DCS camera.
    """
    def __init__(self, env, split='train', seed=0, background=True):
        super().__init__(env, split=split, seed=seed, background=background)
        self.torso_id = self.model.body('torso').id
        self.mounts = []
        for agent, partition in zip(env.possible_agents, env.agent_action_partitions):
            names = [self.native.model.joint(int(self.native.model.actuator_trnid[int(n.act_ids), 0])).name
                     for n in partition]
            hips = [name for name in names if name.startswith('hip_')]
            assert len(hips) == 1
            hip = hips[0]
            body = int(self.model.jnt_bodyid[self.model.joint(hip).id])
            # Rest installation geometry only, no changing limb position lookup.
            mount = self.model.body_pos[body].copy()*2
            mount[2] = -.12
            angle = np.arctan2(mount[1], mount[0])+np.pi/2
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c,-s,0],[s,c,0],[0,0,1.]])
            self.mounts.append((agent, hip, mount, rotation))

    def pixels(self):
        base_pos = self.data.cam_xpos[0].copy()
        base_mat = self.data.cam_xmat[0].copy().reshape(3,3)
        torso = self.data.xpos[self.torso_id].copy()
        body_rotation = self.data.xmat[self.torso_id].reshape(3,3).copy()
        offset = base_pos-self.data.subtree_com[1]
        images = []
        try:
            for _, _, mount, rotation in self.mounts:
                rig = body_rotation@rotation
                self.data.cam_xpos[0] = torso+body_rotation@mount+rig@offset*.5
                self.data.cam_xmat[0] = (rig@base_mat).reshape(-1)
                self.renderer.update_scene(self.data, camera=0)
                images.append(self.renderer.render().copy())
        finally:
            self.data.cam_xpos[0] = base_pos
            self.data.cam_xmat[0] = base_mat.reshape(-1)
        assert bytes(self.native.model.mat_rgba) == self.original
        return np.stack(images)

    def specification(self):
        return dict(kind='fixed torso-mounted local camera rig', image_shape=[4,64,64,3],
                    mounts=[dict(agent=a, hip_joint=h, torso_relative_lookat=m.tolist())
                            for a,h,m,_ in self.mounts],
                    radius_scale=.5, fovy=float(self.model.cam_fovy[0]),
                    perturbations='same archived DCS random camera process in each rotated mount frame',
                    appearance='shared video/color, one advancement per native parallel step',
                    prior='known actuator partition and fixed camera identity; no joint tracking',
                    limits=['local fields can overlap and include partners',
                            'mount camera follows torso pose; explicit rendering-interface prior',
                            'not an official DCS local-camera benchmark'])
