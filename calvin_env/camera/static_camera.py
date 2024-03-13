import numpy as np
import pybullet as p

from calvin_env.camera.camera import Camera

def get_unique_id(pixel):
    if (pixel >= 0):
        obUid = pixel & ((1 << 24) - 1)
        linkIndex = (pixel >> 24) - 1
        if obUid in [0,1,2,3,4]:
            return np.uint8(obUid+1)
        elif obUid == 5:
            return np.uint8(5 + 1 + linkIndex)
        elif obUid == 6:
            return np.uint8(0)
    else:
        return np.uint8(0)

convert_to_known_bodies = np.vectorize(lambda x: get_unique_id(x))

class StaticCamera(Camera):
    def __init__(
        self,
        fov,
        aspect,
        nearval,
        farval,
        width,
        height,
        look_at,
        look_from,
        up_vector,
        cid,
        name,
        robot_id=None,
        objects=None,
    ):
        """
        Initialize the camera
        Args:
            argument_group: initialize the camera and add needed arguments to argparse

        Returns:
            None
        """
        self.nearval = nearval
        self.farval = farval
        self.fov = fov
        self.aspect = aspect
        self.look_from = look_from
        self.look_at = look_at
        self.up_vector = up_vector
        self.width = width
        self.height = height
        self.viewMatrix = p.computeViewMatrix(
            cameraEyePosition=look_from, cameraTargetPosition=look_at, cameraUpVector=self.up_vector
        )
        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov=fov, aspect=aspect, nearVal=self.nearval, farVal=self.farval
        )
        self.cid = cid
        self.name = name

    def set_position_from_gui(self):
        info = p.getDebugVisualizerCamera(physicsClientId=self.cid)
        look_at = np.array(info[-1])
        dist = info[-2]
        forward = np.array(info[5])
        look_from = look_at - dist * forward
        self.viewMatrix = p.computeViewMatrix(
            cameraEyePosition=look_from, cameraTargetPosition=look_at, cameraUpVector=self.up_vector
        )
        look_from = [float(x) for x in look_from]
        look_at = [float(x) for x in look_at]
        return look_from, look_at

    def render(self):
        image = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projectionMatrix,
            physicsClientId=self.cid,
        )
        rgb_img, depth_img = self.process_rgbd(image, self.nearval, self.farval)
        return rgb_img, depth_img

    def render_segmentation(self):
        image = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projectionMatrix,
            physicsClientId=self.cid,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            
            
        )
        
        # # Get the total number of bodies
        # num_bodies = p.getNumBodies()
        # for i in range(num_bodies):
        #     # Get body info
        #     body_info = p.getBodyInfo(i)
        #     print(f"Body ID: {i}, Body info: {body_info}")
        #     # Get the number of joints (links) for this body
        #     num_joints = p.getNumJoints(i)
        #     print(f"Number of joints (links) for body {i}: {num_joints}")
        #     for j in range(num_joints):
        #         # Get joint info
        #         joint_info = p.getJointInfo(i, j)
        #         print(f"Joint ID: {j}, Joint info: {joint_info}")
                
        (width, height, rgbPixels, depthPixels, segmentationMaskBuffer) = image
        # rgb = np.reshape(rgbPixels, (height, width, 4))
        # rgb_img = rgb[:, :, :3]
        # depth_buffer = np.reshape(depthPixels, [height, width])
        # depth = self.z_buffer_to_real_distance(z_buffer=depth_buffer, far=self.nearval, near=self.farval)
        
        img = np.reshape(segmentationMaskBuffer, (height, width))
        img = convert_to_known_bodies(img)
        return img
