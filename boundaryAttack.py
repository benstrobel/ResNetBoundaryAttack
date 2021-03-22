import numpy as np
import mxnet as mx

def perpendicular_vector(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])

class BoundaryAttack:

    def __init__(self, original_img=None, target_img=None):
        if original_img is None or target_img is None:
            raise NotImplementedError("Not Implemented Yet, pass both Original Image and Target Image")
        elif original_img is None != target_img is None:
            raise AttributeError("You need to pass both original and target image to use targeted or neither to use "
                                 "untargeted mode")
        else:
            self.orig_img = original_img
            self.delta = target_img - original_img
        return

    def step(self):
        return

    def __firstPartStep(self):
        return

    def __secondPartStep(self):
        return

    def getCurrentImg(self):
        return self.orig_img + self.delta

    def getCurrentDelta(self):
        return self.delta
