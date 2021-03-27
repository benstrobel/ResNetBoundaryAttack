import numpy as np
import mxnet as mx


def crossProduct(ndarray0, ndarray1):
    return mx.nd.array(np.cross(ndarray0.asnumpy(),ndarray1.asnumpy(), axisa=1, axisb=1,axisc=1))

def normalize(ndarray):
    norm = mx.nd.norm(ndarray)
    return mx.nd.divide(ndarray,norm)

def dist(ndarray):
    return mx.nd.norm(ndarray).asnumpy()[0]


class BoundaryAttack:

    def __init__(self, original_img=None, target_img=None, evaluate=None):
        if original_img is None and target_img is None and evaluate is None:
            raise NotImplementedError("Not Implemented yet, pass Original Image, Target Image and evaluate function")
        elif original_img is None != target_img is None != evaluate is None:
            raise AttributeError(
                "You need to pass original, target image and evaluate function to use targeted or neither to use "
                "untargeted mode")
        else:
            self.orig_img = original_img
            self.delta = target_img - original_img
            self.eval = evaluate
            self.orig_class = self.eval(original_img)
            self.target_class = self.eval(target_img)
            self.alpha = 0.1
            self.beta = 0.1
            self.firstStepSuccess = []
            self.secondStepSuccess = []
            self.successProbabilityAfterStep1 = -1
            self.successProbabilityAfterStep2 = -1
            self.stepCounter = 0
            sanity = self.eval(original_img+self.delta)
            if sanity != self.target_class:
                print("Sanity Check failed: Is: " + str(sanity) + " Should: " + str(self.orig_class))
        return

    def step(self):
        firstStepSuc = None
        secondStepSuc = None
        previousDelta = self.delta
        distance = dist(self.delta)
        self.__firstPartStep()
        current_class = self.eval(self.orig_img + self.delta)
        if current_class == self.target_class:
            self.__secondPartStep()
            new_distance = dist(self.delta)
            if new_distance > distance:
                self.delta = previousDelta
                return
            firstStepSuc = True
            secondStepSuc = self.eval(self.orig_img + self.delta) == self.target_class
        else:
            firstStepSuc = False
            self.delta = previousDelta
            new_distance = dist(self.delta)
        self.firstStepSuccess.append(firstStepSuc)
        self.secondStepSuccess.append(secondStepSuc)
        self.__tuneHyperParameter()
        print("Step " + str('{:<3}'.format(self.stepCounter)) + " complete (" + str(firstStepSuc)[0] + ","
              + str(secondStepSuc)[0] + ") Alpha: " + str('{:<25}'.format(self.alpha)) + " Beta: "
              + str('{:<25}'.format(self.beta)) + " 1stSuccProb: "
              + str('{:<4}'.format(self.successProbabilityAfterStep1*100)) + "% 2ndSuccProb: "
              + str('{:<4}'.format(self.successProbabilityAfterStep2*100)) + "% L2Distance: " + str(new_distance))
        self.stepCounter = self.stepCounter+1
        return

    def __firstPartStep(self):
        randomarray = mx.ndarray.random.uniform(-1, 1, self.orig_img.shape)
        a = crossProduct(randomarray, self.delta)
        orth_perturbation = normalize(a)
        self.delta = self.delta + orth_perturbation * self.alpha * np.random.normal()
        self.cutUnderAndOverflow()
        return

    def __secondPartStep(self):
        self.delta = self.delta + normalize(-self.delta) * self.beta * np.random.normal()  # Random Number from
        self.cutUnderAndOverflow()
        return

    def __tuneHyperParameter(self):
        firstStepSuccChanged = False
        secondStepSuccChanged = False

        # Making sure only the last 10 Steps are getting taken into account for the success rates

        if len(self.firstStepSuccess) > 10:
            self.firstStepSuccess = self.firstStepSuccess[1:]
            firstStepSuccChanged = True
        if len(self.secondStepSuccess) > 10:
            self.secondStepSuccess = self.secondStepSuccess[1:]
            secondStepSuccChanged = True

        # Enable Hyperparemeter Tuning Within the First Ten Iterations
        if len(self.firstStepSuccess) < 10:
            firstStepSuccChanged = True
        if len(self.secondStepSuccess) < 10:
            secondStepSuccChanged = True

        # Calculating the respective the success rates and applying hyperparameter adjustments

        self.successProbabilityAfterStep1 = self.firstStepSuccess.count(True) / len(self.firstStepSuccess)

        if firstStepSuccChanged:
            if abs(self.successProbabilityAfterStep1 - 0.5) > 0.15:
                if self.successProbabilityAfterStep1 == 0:
                    self.successProbabilityAfterStep1 = 0.01
                self.alpha = self.alpha * (self.successProbabilityAfterStep1 / 0.5)

        self.successProbabilityAfterStep2 = self.secondStepSuccess.count(True) / len(self.firstStepSuccess)

        if secondStepSuccChanged:
            if abs(self.successProbabilityAfterStep2 - 0.5) > 0.15:
                if self.successProbabilityAfterStep2 == 0:
                    self.successProbabilityAfterStep2 = 0.01
                self.beta = self.beta * (self.successProbabilityAfterStep2 / 0.5)

        return

    def cutUnderAndOverflow(self):
        min_diff = mx.nd.broadcast_minimum(self.orig_img + self.delta, mx.nd.full(self.orig_img.shape, 0))
        self.delta = self.delta - min_diff
        max_diff = mx.nd.broadcast_maximum(self.orig_img + self.delta - mx.nd.full(self.orig_img.shape, 1),
                                           mx.nd.full(self.orig_img.shape, 0))
        self.delta = self.delta - max_diff
        return

    def getCurrentImg(self):
        return self.orig_img + self.delta

    def getCurrentDelta(self):
        return self.delta

    def getCurrentStep(self):
        return self.stepCounter

    def getCurrentAlpja(self):
        return self.alpha