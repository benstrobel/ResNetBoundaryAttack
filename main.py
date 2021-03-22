from resnetWrapper import preprocess
from resnetWrapper import process
from imageNetLabelDict import labelDict
from boundaryAttack import BoundaryAttack
import mxnet as mx
from matplotlib import image
from matplotlib import pyplot
import os
import random

# For Reproducibility
random.seed(1337)

sealionList= os.listdir("D:\\Dataset\\part-of-imagenet-master\\partial_imagenet\\sealion\\Images\\")
forkLiftList= os.listdir("D:\\Dataset\\part-of-imagenet-master\\partial_imagenet\\forklift\\Images\\")

sealionImg = image.imread("D:\\Dataset\\part-of-imagenet-master\\partial_imagenet\\sealion\\Images\\" + sealionList[random.randrange(0,len(sealionList))]) #sealion
forkLiftImg = image.imread("D:\\Dataset\\part-of-imagenet-master\\partial_imagenet\\forklift\\Images\\" + forkLiftList[random.randrange(0,len(forkLiftList))]) #forklift
sealionImgPreprocessed = preprocess(mx.nd.array(sealionImg))
forkLiftImgPreprocessed = preprocess(mx.nd.array(forkLiftImg))

img = sealionImg

preprocessed = preprocess(mx.nd.array(img))
result_label_index = process(preprocessed)
print("Result Class: " + str(result_label_index) + " " + labelDict[result_label_index])
boundaryAttack = BoundaryAttack(preprocessed, forkLiftImgPreprocessed)