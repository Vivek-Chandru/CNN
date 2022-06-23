# Neural style transfer using Convolutional neural networks

A code to perform style transfer from a style image onto a content image. The code is based on the neural style transfer technique mentioned in the book **Generative deep learning** by **David Foster**

The code builds a StyleContentModel from the pre-trained and available VGG19 CNN. The results form the intermediate layers of VGG19 are used to evaluate style and content losses which can be optimized to generate an image with the style image imposed on the content image. 

Example: style_transfer.png 
where:
left image - style image
right image - content image
midlle image - result image

>**Note:** The code is written with tensorflow 2.31




