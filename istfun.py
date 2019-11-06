# imports
import numpy as np
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b
from PIL import Image

IMAGENET_MEAN_RGB_VALUES = [123.68, 116.779, 103.939]


def convert_to_image_array(img):
    # Data normalization and reshaping from RGB to BGR
    image_array = np.asarray(img, dtype="float32")
    image_array = np.expand_dims(image_array, axis=0)
    image_array[:, :, :, 0] -= IMAGENET_MEAN_RGB_VALUES[2]
    image_array[:, :, :, 1] -= IMAGENET_MEAN_RGB_VALUES[1]
    image_array[:, :, :, 2] -= IMAGENET_MEAN_RGB_VALUES[0]
    return  image_array[:, :, :, ::-1]
    
    
    
def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))
    

def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram
  
def compute_style_loss(style, combination,IMAGE_HEIGHT,IMAGE_WIDTH,CHANNELS):
    style = gram_matrix(style)
    combination = gram_matrix(combination)
    size = IMAGE_HEIGHT * IMAGE_WIDTH
    return backend.sum(backend.square(style - combination)) / (4. * (CHANNELS ** 2) * (size ** 2))
    


def total_variation_loss(x,IMAGE_HEIGHT,IMAGE_WIDTH,TOTAL_VARIATION_LOSS_FACTOR):
    a = backend.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, 1:, :IMAGE_WIDTH-1, :])
    b = backend.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, :IMAGE_HEIGHT-1, 1:, :])
    return backend.sum(backend.pow(a + b, TOTAL_VARIATION_LOSS_FACTOR))
  

def convert_to_pil_image(img):
    if type(img) ==  np.ndarray:
        if img.dtype != np.uint8:
            img = (img*255).astype(np.unit8) 
        img = Image.fromarray(img)  
    return img    
      
def image_style_transfer(content,style,IMAGE_HEIGHT = 500, IMAGE_WIDTH = 500, 
                         CHANNELS = 3,ITERATIONS = 10,CONTENT_WEIGHT = 0.02,STYLE_WEIGHT = 4.5,
                         TOTAL_VARIATION_WEIGHT = 0.995,TOTAL_VARIATION_LOSS_FACTOR = 1.25):
    # Model
    content = convert_to_pil_image(content)
    style = convert_to_pil_image(style)
    content_image = content.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    style_image = style.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

    input_image_array = convert_to_image_array(content_image)
    style_image_array = convert_to_image_array(style_image)
    
    input_image = backend.variable(input_image_array)
    style_image = backend.variable(style_image_array)
    
    combination_image = backend.placeholder((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    
    input_tensor = backend.concatenate([input_image,style_image,combination_image], axis=0)
    model = VGG16(input_tensor=input_tensor, include_top=False)
    
    layers = dict([(layer.name, layer.output) for layer in model.layers])

    content_layer = "block2_conv2"
    layer_features = layers[content_layer]
    content_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]

    loss = backend.variable(0.)
    loss += CONTENT_WEIGHT * content_loss(content_image_features,
                                          combination_features)
     
    style_layers = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]
    
    for layer_name in style_layers:
        layer_features = layers[layer_name]
        style_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        style_loss = compute_style_loss(style_features, combination_features,IMAGE_HEIGHT,IMAGE_WIDTH,CHANNELS)
        loss += (STYLE_WEIGHT / len(style_layers)) * style_loss
   
    loss += TOTAL_VARIATION_WEIGHT * total_variation_loss(combination_image,IMAGE_HEIGHT,IMAGE_WIDTH,TOTAL_VARIATION_LOSS_FACTOR)
    
    outputs = [loss]
    outputs += backend.gradients(loss, combination_image)
    
    def evaluate_loss_and_gradients(x):
        x = x.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
        outs = backend.function([combination_image], outputs)([x])
        loss = outs[0]
        gradients = outs[1].flatten().astype("float64")
        return loss, gradients

    class Evaluator:

        def loss(self, x):
            loss, gradients = evaluate_loss_and_gradients(x)
            self._gradients = gradients
            return loss

        def gradients(self, x):
            return self._gradients
    
    evaluator = Evaluator()
    x = np.random.uniform(0, 255, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)) - 128.

    for i in range(ITERATIONS):
        x, loss, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.gradients, maxfun=20)
        print("Iteration %d completed with loss %d" % (i, loss))

    x = x.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    x = x[:, :, ::-1]
    x[:, :, 0] += IMAGENET_MEAN_RGB_VALUES[2]
    x[:, :, 1] += IMAGENET_MEAN_RGB_VALUES[1]
    x[:, :, 2] += IMAGENET_MEAN_RGB_VALUES[0]
    x = np.clip(x, 0, 255).astype("uint8")

    output_image = convert_to_image_array(x).resize(content_image.size)
    return output_image
