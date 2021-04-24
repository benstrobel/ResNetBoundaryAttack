# Mit mehr Instanzen loopen -> Nur bis 200 Schritte testen
# Alle Parameter in ne config rein
# Bei vielen Ausführungen messbare Ergebnisse

from resnetWrapper import ResNet
from imageNetLabelDict import labelDict
from boundaryAttack import BoundaryAttack
import mxnet as mx
from matplotlib import image
from matplotlib import pyplot
from PIL import Image
import os
import random
import numpy as np
import datetime
import sys


def render_as_image(a):
    img = a.asnumpy()  # convert to numpy array
    img = img.transpose((1, 2, 0))  # Move channel to the last dimension
    img = np.multiply(img, 255)
    img = img.astype(np.uint8)  # use uint8 (0-255)

    pyplot.imshow(img)
    pyplot.show()
    return img

def save_image(name, date, img):
    pyplot.imsave(os.getcwd() + "\\Result\\" + date.strftime("%Y%m%d%H%M%S") + "\\" + name + ".png", img,
                  format="png")

def save_figure(name, date):
    pyplot.savefig(os.getcwd() + "\\Result\\" + date.strftime("%Y%m%d%H%M%S") + "\\" + name + ".png")

def getScore(instance):
    if instance.lastStep is None or instance.lastStep == 0:
        return instance.boundaryAttack.getCurrentDist()
    else:
        return instance.boundaryAttack.getCurrentDist() * pow(instance.lastStep, 2)

def getInstanceWithLowestDist(instanceList):
    lowestCalculatedStep = sys.maxsize
    for x in instanceList:
        if x.lastStep < lowestCalculatedStep:
            lowestCalculatedStep = x.lastStep

    if lowestCalculatedStep == sys.maxsize or lowestCalculatedStep is None or lowestCalculatedStep == 0:
        return instanceList[0]

    lowestCalculatedStep = lowestCalculatedStep - 1

    currentLowest = instanceList[0]
    currentLowestScore = instanceList[0].distance_list[lowestCalculatedStep]

    if len(instanceList) > 1:
        for x in instanceList:
            currentScore = x.distance_list[lowestCalculatedStep]
            if currentScore < currentLowestScore:
                currentLowest = x
                currentLowestScore = currentScore
    return currentLowest


# Setting Mean And Std Array
mean_r = mx.nd.full((1, 224, 224), 0.485)
mean_g = mx.nd.full((1, 224, 224), 0.456)
mean_b = mx.nd.full((1, 224, 224), 0.406)
mean = mx.nd.concat(mean_r, mean_g, mean_b, dim=0)

std_r = mx.nd.full((1, 224, 224), 0.229)
std_g = mx.nd.full((1, 224, 224), 0.224)
std_b = mx.nd.full((1, 224, 224), 0.225)
std = mx.nd.concat(std_r, std_g, std_b, dim=0)

# For Reproducibility
random.seed(1337)

pathToCategories = "F:\\DatasetDownloader\\partial_imagenet\\"
categoryList = os.listdir(pathToCategories)

below_convergence_limit_counter = 0
convergence_limit = 0.001
abort_after_step = 200


class AttackInstance:

    def __init__(self, origin_img, target_img=None):
        self.resnet = ResNet()

        self.origin_preprocessed = self.resnet.preprocess(mx.nd.array(origin_img))
        if target_img is not None:
            self.target_preprocessed = self.resnet.preprocess(mx.nd.array(target_img))
        else:
            self.target_preprocessed = None
        result_label_index = self.resnet.process(self.origin_preprocessed)
        print("Result Class: " + str(result_label_index) + " " + labelDict[result_label_index])
        self.boundaryAttack = BoundaryAttack(self.origin_preprocessed, self.target_preprocessed, self.resnet.process)
        if self.boundaryAttack.target_class is not None:
            print("Target Class: " + str(self.boundaryAttack.target_class) + " " + labelDict[
                self.boundaryAttack.target_class])
        render_as_image(self.origin_preprocessed[0] * std + mean)
        img = render_as_image((self.origin_preprocessed + self.boundaryAttack.getCurrentDelta())[0] * std + mean)
        self.distance_list = []
        self.alpha_list = []
        self.beta_list = []
        self.lastStep = 0
        self.date = datetime.datetime.now()
        os.mkdir(os.getcwd()+"\\Result\\" + self.date.strftime("%Y%m%d%H%M%S"))
        save_image("start", self.date, img)
        return

    def step(self):
        self.boundaryAttack.step()
        if self.lastStep == 0 or self.lastStep is None or self.lastStep < self.boundaryAttack.getCurrentStep():
            self.lastStep = self.boundaryAttack.getCurrentStep()
            self.distance_list.append(self.boundaryAttack.getCurrentDist())
            self.alpha_list.append(self.boundaryAttack.getCurrentAlpha())
            self.beta_list.append(self.boundaryAttack.getCurrentBeta())
        if self.boundaryAttack.getCurrentAlpha() < convergence_limit:
            self.below_convergence_limit_counter = self.below_convergence_limit_counter + 1
        else:
            self.below_convergence_limit_counter = 0

    def finish(self):
        img = render_as_image((self.origin_preprocessed + self.boundaryAttack.getCurrentDelta())[0] * std + mean)

        print("Finished Adversarial Sample within " + str(self.boundaryAttack.stepCounter) + " Steps and "
              + str(self.resnet.forward_counter) + " Forward Passes (" + str(self.boundaryAttack.firstSuccCount) + "," +
              str(self.boundaryAttack.secondSuccCount) + "," + str(self.boundaryAttack.bothSuccCount) + ")")
        pyplot.plot(self.distance_list)
        pyplot.ylabel("L2-Distance")
        pyplot.xlabel("Step")
        save_figure("distance", self.date)
        pyplot.show()

        pyplot.plot(self.alpha_list)
        pyplot.ylabel("Alpha")
        pyplot.xlabel("Step")
        save_figure("alpha", self.date)
        pyplot.show()

        pyplot.plot(self.beta_list)
        pyplot.ylabel("Beta")
        pyplot.xlabel("Step")
        save_figure("beta", self.date)
        pyplot.show()

        save_image("result", self.date, img)

class Runner:
    def __init__(self):
        sourceCategoryIndex = random.randrange(0, len(categoryList))
        targetCategoryIndex = None
        while(targetCategoryIndex == None):
            index = random.randrange(0, len(categoryList))
            if index != sourceCategoryIndex:
                targetCategoryIndex = index

        sourcePath = pathToCategories + categoryList[sourceCategoryIndex] + "\\Images\\"
        targetPath = pathToCategories + categoryList[targetCategoryIndex] + "\\Images\\"

        soruceImageList = os.listdir(sourcePath)
        targetImageList = os.listdir(targetPath)

        sourceImage = image.imread(sourcePath + soruceImageList[random.randrange(0, len(soruceImageList))])

        targetImage = []
        self.instanceList = []
        self.instances = 4

        for x in range(self.instances):
            targetImage.append(image.imread(targetPath + targetImageList[random.randrange(0, len(targetImageList))]))

        for x in range(self.instances):
            self.instanceList.append(AttackInstance(sourceImage, targetImage[x]))

        self.currentInstance = getInstanceWithLowestDist(self.instanceList)


    def run_scheuduled(self):
        self.total_steps = 0
        self.stepList = []
        while self.total_steps <= abort_after_step and below_convergence_limit_counter < 5:
            cur = self.currentInstance
            if self.total_steps % (25 * self.instances) == 0:
                print("Performing " + str(2*self.instances) + " Steps for every Instance")
                for x in self.instanceList:
                    for i in range(2*self.instances):
                        curStep = x.lastStep
                        while curStep == x.lastStep:
                            x.step()
                            self.stepList.append(self.instanceList.index(x))
                self.currentInstance = getInstanceWithLowestDist(self.instanceList)
            if cur != self.currentInstance:
                print("Swapped Instance " + str(self.instanceList.index(cur)) + " -> " + str(self.instanceList.index(self.currentInstance)))
            self.currentInstance.step()
            self.stepList.append(self.instanceList.index(self.currentInstance))
            self.total_steps = self.total_steps + 1
        self.currentInstance.finish()

        self.showAndSaveResults()

    def run_unscheuduled(self):
        self.current_steps = 0
        for instance in self.instanceList:
            print("Starting with next instance")
            while self.current_steps <= abort_after_step:
                instance.step()
            self.current_steps = 0

        self.showAndSaveResults()

    def showAndSaveResults(self):
        for x in self.instanceList:
            pyplot.plot(x.distance_list)
        pyplot.ylabel("L2-Distance")
        pyplot.xlabel("Step")
        save_figure("distance-all", self.currentInstance.date)
        pyplot.show()


runner = Runner()
runner.run_unscheuduled()
runner.run_scheuduled()

