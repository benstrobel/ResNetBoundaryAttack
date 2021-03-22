from resnetWrapper import preprocess
from resnetWrapper import process
from imageNetLabelDict import labelDict
import mxnet as mx
from matplotlib import image
from matplotlib import pyplot


img = image.imread("D:\\Dataset\\part-of-imagenet-master\\partial_imagenet\\sealion\\Images\\n02077923_15.jpg") #sealion0
pyplot.imshow(img)
pyplot.show()
preprocessed = preprocess(mx.nd.array(img))
result_label_index = process(preprocessed)
print("Result Class: " + str(result_label_index) + " " + labelDict[result_label_index])
