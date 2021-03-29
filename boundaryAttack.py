import numpy as np
import mxnet as mx
import datetime
import random


def crossProduct(ndarray0, ndarray1):
    return mx.nd.array(np.cross(ndarray0.asnumpy(),ndarray1.asnumpy(), axisa=1, axisb=1,axisc=1))

def normalize(ndarray):
    norm = mx.nd.norm(ndarray)
    return mx.nd.divide(ndarray,norm)

def dist(ndarray):
    return mx.nd.norm(ndarray).asnumpy()[0]

def orth_step(ndarray):
    rgb_rnd_index = random.randint(0,2)
    x_rnd_index = random.randint(0,223)
    y_rnd_index = random.randint(0,223)
    delta_value = ndarray.asnumpy()[0][rgb_rnd_index][x_rnd_index][y_rnd_index]
    sum = np.sum(ndarray.asnumpy()) - delta_value
    orth_step = mx.nd.full(ndarray.shape, 1)
    orth_step[0][rgb_rnd_index][x_rnd_index][y_rnd_index] = sum / delta_value
    return orth_step

class BoundaryAttack:

    def __init__(self, original_img=None, target_img=None, evaluate=None):
        self.alpha = 0.1
        self.beta = 0.1
        self.firstStepSuccess = []
        self.secondStepSuccess = []
        self.successProbabilityAfterStep1 = -1
        self.successProbabilityAfterStep2 = -1
        self.stepCounter = 0
        self.firstSuccCount = 0
        self.secondSuccCount = 0
        self.bothSuccCount = 0
        if original_img is not None and target_img is None and evaluate is not None:        # Untargeted Mode
            print("Untargeted Mode")
            self.orig_img = original_img
            self.eval = evaluate
            self.orig_class = self.eval(original_img)
            self.delta = None
            self.criteriaFct = lambda: self.eval(original_img + self.delta) != self.orig_class
            self.target_class = None
            while self.delta is None or not self.criteriaFct():
                self.delta = mx.ndarray.random.uniform(0, 1, original_img.shape)
            print("Found starting deviation")
        elif original_img is not None and target_img is not None and evaluate is not None:  # Targeted Mode
            print("Targeted Mode")
            self.orig_img = original_img
            self.delta = target_img - original_img
            self.eval = evaluate
            self.orig_class = self.eval(original_img)
            self.target_class = self.eval(target_img)
            self.criteriaFct = lambda: self.eval(original_img + self.delta) == self.target_class
        else:
            raise AttributeError(
                "You need to pass original, target image and evaluate function to use targeted or just original and "
                "evalute to use untargeted mode")

        if not self.criteriaFct():
            print("Sanity Check failed")
        return

    def step(self):
        firstStepSuc = None
        secondStepSuc = None
        previousDelta = self.delta
        distance = dist(self.delta)
        self.__firstPartStep()
        if self.criteriaFct():
            self.__secondPartStep()
            new_distance = dist(self.delta)
            if new_distance > distance:
                self.delta = previousDelta
                return
            firstStepSuc = True
            secondStepSuc = self.criteriaFct()
            if not secondStepSuc:
                self.delta = previousDelta
                new_distance = distance
        else:
            firstStepSuc = False
            self.delta = previousDelta
            new_distance = distance
        self.firstStepSuccess.append(firstStepSuc)
        if secondStepSuc is not None:
            self.secondStepSuccess.append(secondStepSuc)
        self.__tuneHyperParameter()

        print(str(datetime.datetime.now()) + " Step " + str('{:<3}'.format(self.stepCounter)) + " complete (" +
              str(firstStepSuc)[0] + "," + str(secondStepSuc)[0] + ") Alpha: " + str('{:<25}'.format(self.alpha))
              + " Beta: " + str('{:<25}'.format(self.beta)) + " 1stSuccProb: "
              + str('{:<4}'.format(self.successProbabilityAfterStep1*100)) + "% 2ndSuccProb: "
              + str('{:<4}'.format(self.successProbabilityAfterStep2*100)) + "% L2Distance: " + str(new_distance))
        self.stepCounter = self.stepCounter+1
        return

    def __firstPartStep(self):
        random_array = orth_step(self.delta)
        orth_perturbation = normalize(random_array)
        first_step = orth_perturbation * self.alpha * np.random.normal()
        self.delta = self.delta + first_step
        #self.cutUnderAndOverflow()
        return

    def __secondPartStep(self):
        self.delta = self.delta + normalize(-self.delta) * self.beta * np.random.normal()
        #self.cutUnderAndOverflow()
        return

    def __tuneHyperParameter(self):
        firstStepSuccChanged = False
        secondStepSuccChanged = False

        # Making sure only the last 10 Steps are getting taken into account for the success rates
        list_max_elements = 10

        if len(self.firstStepSuccess) > list_max_elements:
            self.firstStepSuccess = self.firstStepSuccess[1:]
            firstStepSuccChanged = True
        if len(self.secondStepSuccess) > list_max_elements:
            self.secondStepSuccess = self.secondStepSuccess[1:]
            secondStepSuccChanged = True

        # Enable Hyperparemeter Tuning Within the First Ten Iterations
        if len(self.firstStepSuccess) < list_max_elements:
            firstStepSuccChanged = True
        if len(self.secondStepSuccess) < list_max_elements:
            secondStepSuccChanged = True

        # Calculating the respective success rates and applying hyperparameter adjustments

        self.successProbabilityAfterStep1 = self.firstStepSuccess.count(True) / len(self.firstStepSuccess)

        if firstStepSuccChanged:
            self.firstSuccCount = self.firstSuccCount + 1
            # Restricting by how much alpha can change within one adjustment
            successProbabilityAfterStep1 = min(max(self.successProbabilityAfterStep1,0.35),0.65)
            self.alpha = self.alpha * (successProbabilityAfterStep1 / 0.5)

        self.successProbabilityAfterStep2 = self.secondStepSuccess.count(True) / len(self.firstStepSuccess)

        if secondStepSuccChanged:
            self.secondSuccCount = self.secondSuccCount + 1
            # Restricting by how much beta can change within one adjustment
            successProbabilityAfterStep2 = min(max(self.successProbabilityAfterStep2,0.35),0.65)
            self.beta = self.beta * (successProbabilityAfterStep2 / 0.5)

        if firstStepSuccChanged and secondStepSuccChanged:
            self.bothSuccCount = self.bothSuccCount + 1

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

    def getCurrentAlpha(self):
        return self.alpha

    def getCurrentBeta(self):
        return self.beta

    def getCurrentDist(self):
        return dist(self.delta)