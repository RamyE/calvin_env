import numpy as np
import pybullet as p

from calvin_env.camera.camera import Camera


"""
For each pixels the visible object unique id. If ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX is used, the segmentationMaskBuffer combines the object unique id and link index as follows:
value = objectUniqueId + (linkIndex+1)<<24. See example. https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/segmask_linkindex.py

So for a free floating body without joints/links, the segmentation mask is equal to its body unique id, since its link index is -1.
https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit
"""
def get_unique_id(pixel, numBodies):
    assert numBodies in [6, 7, 8]
    if (pixel >= 0):
        obUid = pixel & ((1 << 24) - 1)
        linkIndex = (pixel >> 24) - 1
        # print(obUid, linkIndex)
        if numBodies == 6:
            # 0 is the robot, make it 1
            if obUid == 0:
                return np.uint8(1)
            # 1, 2, 3 are the red, blue, pink cubes, make them 2, 3, 4
            if obUid in [1,2,3]:
                assert linkIndex == -1
                return np.uint8(obUid+1)
            elif obUid == 4:
                return np.uint8(5 + 1 + linkIndex)
            # 5 is the background, make it 0
            elif obUid == 5:
                return np.uint8(0)
            else:
                raise ValueError(f"Unknown object {obUid}")
        elif numBodies in [7, 8]:
            # 0 is the robot, make it 1
            if obUid == 0:
                return np.uint8(1)
            # 2, 3, 4 are the red, blue, pink cubes, make them 2, 3, 4
            if obUid in [2,3,4]:
                assert linkIndex == -1
                return np.uint8(obUid)
            elif obUid == 5:
                return np.uint8(5 + 1 + linkIndex)
            # 6 is the background, make it 0
            elif obUid == 6:
                return np.uint8(0)
            elif obUid in [1, 7]: # 1 and 7 are link0 (bottom part of the robot)
                return np.uint8(1)
            else:
                raise ValueError(f"Unknown object {obUid}")
    else:
        return np.uint8(0)

convert_to_known_bodies = np.vectorize(get_unique_id)

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

    def render_segmentation(self, numBodies):
        # print("Num bodies", numBodies)
        image = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projectionMatrix,
            physicsClientId=self.cid,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        )     
        
        (width, height, rgbPixels, depthPixels, segmentationMaskBuffer) = image
        
        img = np.reshape(segmentationMaskBuffer, (height, width))
        # print(set(np.array(segmentationMaskBuffer).flatten()))
        # print(len(set(np.array(segmentationMaskBuffer).flatten())))
        # print({x: np.sum(img == x) for x in set(np.array(segmentationMaskBuffer).flatten())})
        img = convert_to_known_bodies(img, numBodies)
        return img
