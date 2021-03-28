from resnetWrapper import ResNet
from imageNetLabelDict import labelDict
from boundaryAttack import BoundaryAttack
import mxnet as mx
from matplotlib import image
from matplotlib import pyplot
import os
import random
import numpy as np

def render_as_image(a):
    img = a.asnumpy() # convert to numpy array
    img = img.transpose((1, 2, 0))  # Move channel to the last dimension
    img = np.multiply(img,255)
    img = img.astype(np.uint8)  # use uint8 (0-255)

    pyplot.imshow(img)
    pyplot.show()


# Setting Mean And Std Array
mean_r = mx.nd.full((1,224,224), 0.485)
mean_g = mx.nd.full((1,224,224), 0.456)
mean_b = mx.nd.full((1,224,224), 0.406)
mean = mx.nd.concat(mean_r,mean_g,mean_b,dim=0)

std_r = mx.nd.full((1,224,224), 0.229)
std_g = mx.nd.full((1,224,224), 0.224)
std_b = mx.nd.full((1,224,224), 0.225)
std = mx.nd.concat(std_r,std_g,std_b,dim=0)

# For Reproducibility
random.seed(1337)

sealionPath = "D:\\Dataset\\part-of-imagenet-master\\partial_imagenet\\sealion\\Images\\"
forkLiftPath = "D:\\Dataset\\part-of-imagenet-master\\partial_imagenet\\forklift\\Images\\"

sealionList= os.listdir(sealionPath)
forkLiftList= os.listdir(forkLiftPath)

resnet = ResNet()

sealionImg = image.imread(sealionPath + sealionList[random.randrange(0, len(sealionList))])
forkLiftImg = image.imread(forkLiftPath + forkLiftList[random.randrange(0, len(forkLiftList))])
sealionImgPreprocessed = resnet.preprocess(mx.nd.array(sealionImg))
forkLiftImgPreprocessed = resnet.preprocess(mx.nd.array(forkLiftImg))

img = sealionImg

preprocessed = resnet.preprocess(mx.nd.array(img))

result_label_index = resnet.process(preprocessed)
print("Result Class: " + str(result_label_index) + " " + labelDict[result_label_index])

#Replace forkLiftImgPreprocessed with None to use untargeted mode
boundaryAttack = BoundaryAttack(preprocessed, None, resnet.process)
if boundaryAttack.target_class is not None:
    print("Target Class: " + str(boundaryAttack.target_class) + " " + labelDict[boundaryAttack.target_class])

below_convergence_limit_counter = 0
convergence_limit = 0.001

render_as_image(preprocessed[0]*std + mean)
render_as_image((preprocessed+boundaryAttack.getCurrentDelta())[0]*std + mean)

distance_list = []
alpha_list = []
beta_list = []

while boundaryAttack.getCurrentStep() < 400 and below_convergence_limit_counter < 5:
    boundaryAttack.step()
    distance_list.append(boundaryAttack.getCurrentDist())
    alpha_list.append(boundaryAttack.getCurrentAlpha())
    beta_list.append(boundaryAttack.getCurrentBeta())
    if boundaryAttack.getCurrentAlpha() < convergence_limit:
        below_convergence_limit_counter = below_convergence_limit_counter + 1
    else:
        below_convergence_limit_counter = 0

render_as_image((preprocessed+boundaryAttack.getCurrentDelta())[0]*std + mean)
print("Finished Adversarial Sample within " + str(boundaryAttack.stepCounter) + " Steps and " + str(resnet.forward_counter) + " Forward Passes.")
pyplot.plot(distance_list)
pyplot.ylabel("L2-Distance")
pyplot.xlabel("Step")
pyplot.show()

pyplot.plot(alpha_list)
pyplot.ylabel("Alpha")
pyplot.xlabel("Step")
pyplot.show()

pyplot.plot(beta_list)
pyplot.ylabel("Beta")
pyplot.xlabel("Step")
pyplot.show()
