import numpy as np
import mxnet as mx
import math

def perpendicular_vector(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])

class BoundaryAttack:

    def __init__(self, original_img=None, target_img=None, evaluate = None):
        if original_img is None and target_img is None and evaluate is None:
            raise NotImplementedError("Not Implemented Yet, pass Original Image, Target Image and evaluate function")
        elif original_img is None != target_img is None != evaluate is None:
            raise AttributeError("You need to pass original, target image and evaluate function to use targeted or neither to use "
                                 "untargeted mode")
        else:
            self.orig_img = original_img
            self.delta = target_img - original_img
            self.eval = evaluate
            self.orig_class = eval(original_img)
            self.target_class = eval(target_img)
            self.alpha = 0.1
            self.beta = 0.1
            self.firstStepSuccess = []
            self.secondStepSuccess = []
        return

    def step(self):
        previousDelta = self.delta
        self.__firstPartStep()
        if self.eval(self.orig_img + self.delta) == self.target_class:
            self.firstStepSuccess.append(True)
            self.__secondPartStep()
            self.secondStepSuccess.append(self.eval(self.orig_img + self.delta) == self.target_class)
        else:
            self.firstStepSuccess.append(False)
            self.delta = previousDelta
        self.__tuneHyperParameter()
        return

    def __firstPartStep(self):
        x = None # TODO Find any orthogonal vector / array to current self.delta
        self.delta = self.delta + x * self.alpha * np.random.normal()
        return

    def __secondPartStep(self):
        self.delta = self.delta + mx.nd.norm(-self.delta, 2) * self.beta * np.random.normal() # Random Number from Gaussian Distribution
        return

    def __tuneHyperParameter(self):
        if len(self.firstStepSuccess) > 10:
            self.firstStepSuccess = self.firstStepSuccess[1:]
        if len(self.secondStepSuccess) > 10:
            self.secondStepSuccess = self.secondStepSuccess[1:]

        successProbabilityAfterStep1 = self.firstStepSuccess.count(True) / 10
        successProbabilityAfterStep2 = self.secondStepSuccess.count(True) / 10

        if abs(successProbabilityAfterStep1 - 0.5) > 0.15:
            self.alpha = self.alpha * (0.5 / successProbabilityAfterStep1)

        if abs(successProbabilityAfterStep2 - 0.5) > 0.15:
            self.beta = self.beta * (0.5 / successProbabilityAfterStep2)

        return

    def getCurrentImg(self):
        return self.orig_img + self.delta

    def getCurrentDelta(self):
        return self.delta
