#!/usr/bin/python
# coding: utf-8


from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
import PIL.Image
import time

def load_img(path_to_img, max_dim):
    img=tf.io.read_file(path_to_img)
    img=tf.image.decode_image(img, channels=3)
    img=tf.image.convert_image_dtype(image=img,dtype=tf.float32)
    
    m=max(img.shape)
    h=(img.shape[0]/m)*max_dim
    w=(img.shape[1]/m)*max_dim
    
    img=tf.image.resize(img,tf.cast([h,w],tf.int32)#, 
                        #preserve_aspect_ratio=True
                       )
    img=tf.expand_dims(img,0)
    return img
def vgg_layers(layer_names):
    VGG19=tf.keras.applications.VGG19(include_top = False,weights="imagenet")
    out = [VGG19.get_layer(name).output for name in layer_names]
    model=tf.keras.Model(inputs=VGG19.input, outputs=out)
    return model #typ: keras.models.Model
def gram_matrix(input_tensor):
    # linear transformation from (b,y,x,c)-indexed array to (b,y,x)-indexed array
    temp = tf.linalg.einsum('bijc,bijd->bcd', input_tensor,input_tensor)
    shape = tf.shape(input_tensor)
    num_locations = tf.cast(shape[1]*shape[2],tf.float32)
    return temp/(num_locations)
    
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg =  vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False   
    def call(self, inputs):
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs*255)
        outputs = self.vgg(preprocessed_input)
        style_outputs = outputs[:self.num_style_layers]
        content_outputs  = outputs[self.num_style_layers:]
        style_outputs = [gram_matrix(i) for i in style_outputs]
        content_dict = {content_name:value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name:value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content':content_dict, 'style':style_dict}
        

def clip_0_1(image):
    image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)
    return image
def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight, num_style_layers, num_content_layers):
    style_outputs = outputs['style'] #
    content_outputs = outputs['content'] # 
    # (style_targets[name]-style_outputs[name])**2 #
    style_loss = tf.add_n([tf.reduce_sum(
        (style_targets[name]-style_outputs[name])**2) /
                           tf.reduce_prod(tf.cast(tf.shape(style_outputs[name]), tf.float32)) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers / 4
    # same as above #
    content_loss = tf.add_n([tf.reduce_sum((content_targets[name]-content_outputs[name])**2) for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers / 2
    loss = style_loss + content_loss
    return loss

@tf.function()
def train_step(extractor, image, style_targets, content_targets, style_weight, content_weight,
               num_style_layers, num_content_layers, opt , total_variation_weight):
    with tf.GradientTape() as tape:
        outputs = extractor.call(image)
        loss = style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight, num_style_layers, num_content_layers)
        loss += total_variation_weight*tf.image.total_variation(image)
        #
        #print(loss)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)



def main(content_path,style_path, dim, epochs,steps_per_epoch,save_path,total_variation_weight,content_weight,style_weight):
    # change:
    # initialize image (content_image, random, constant)
    # parts of lists in content_layers/style_layers of VGG19
    # tf.optimizer.Adam

    content_image=load_img(max_dim=dim,path_to_img=content_path)
    style_image=load_img(max_dim=dim,path_to_img=style_path)
    vgg19=tf.keras.applications.VGG19(include_top = False,weights="imagenet")

    ####### Change if desired ###########
    content_layers=["block5_conv2"]
    style_layers=["block1_conv1","block2_conv1","block3_conv1","block4_conv1","block5_conv1"]
    
    num_content_layers=len(content_layers)
    num_style_layers=len(style_layers)
    extractor = StyleContentModel(style_layers,content_layers)
    style_targets = extractor.call(style_image)['style']
    content_targets = extractor.call(content_image)['content']
    image = tf.Variable(content_image)

    ####### Change if desired ###########
    opt = tf.optimizers.Adam(learning_rate = 0.02, beta_1 =0.99, epsilon = 1e-1)
    
    start = time.time()
    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(extractor, image, style_targets, content_targets, style_weight, content_weight, num_style_layers, num_content_layers, opt , total_variation_weight)
            #print(".\n")
        #display.clear_output(wait=True)
        #display.display(tensor_to_image(image))
        print("Train step: {}".format(step)," / ",(epochs*steps_per_epoch),end='\r')
    end = time.time()
    print("\nTotal time: {:.1f}".format(end-start))

    tensor_to_image(image).save(save_path)



if __name__ == "__main__":
   
    ap = argparse.ArgumentParser()
    ap.add_argument("content_path",help = "filename of content image")#,dest=content_path)
    ap.add_argument("style_path",help = "filename of style image")#, dest=style_path)
    ap.add_argument("save_path",help = "save file to destination filename")#, dest=style_path)
    
    ap.add_argument("-d", "--dim", required=False, default=512, type=int, help = "dimensions of image output")#,dest=dim)
    ap.add_argument("-e", "--epochs", required=False, default=100, type=int)
    ap.add_argument("-s", "--steps_per_epoch", required=False, default=10, type=int)
    ap.add_argument("-vw", "--total_variation_weight", required=False, default=100, type=int)
    ap.add_argument("-cw", "--content_weight", required=False, default=1.0, type=float)
    ap.add_argument("-sw", "--style_weight", required=False, default=100.0, type=float)
    # ...
    kwargs = vars(ap.parse_args())
    main(**kwargs)
    
