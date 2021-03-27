# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
.. _l-example-simple-usage:

Load and predict with ONNX Runtime and a very simple model
==========================================================

This example demonstrates how to load a model and compute
the output for an input vector. It also shows how to
retrieve the definition of its inputs and outputs.
"""

import os
import onnxruntime as rt
import numpy as np
from mxnet.gluon.data.vision import transforms
from onnxruntime.datasets import get_example
from PIL import Image

# The 'preprocess' method originated from https://github.com/onnx/onnx
def preprocess(img):
    '''
    Preprocessing required on the images for inference with mxnet gluon
    '''
    transform_fn = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform_fn(img)
    img = img.expand_dims(axis=0)  # batchify

    return img

def to_shape(img):
    transform_fn = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    img = transform_fn(img)
    img = img.expand_dims(axis=0)  # batchify

    return img

def back_to_img(ndarray):
    array = np.moveaxis(ndarray[0].asnumpy(), 0, 2)
    array = np.multiply(array, 255)
    print(array.shape)
    return Image.fromarray(array, 'RGB')

def process(preprocessed_img):

    model = get_example(os.getcwd() + "\\models\\vision\\classification\\resnet\\model\\resnet50-v2-7.onnx")
    sess = rt.InferenceSession(model)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    res = sess.run([output_name], {input_name: preprocessed_img.asnumpy()})
    indexOfMax = (res[0].argmax(axis=1))[0]
    return indexOfMax