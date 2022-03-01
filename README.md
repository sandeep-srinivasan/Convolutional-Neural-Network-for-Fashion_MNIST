# Convolutional-Neural-Network-for-Fashion_MNIST-
Implementation of CNN for Fashion-MNIST dataset

Fashion-MNIST dataset consists of 60,000 28x28 grayscale training images of clothing items of one of ten types (shirt, dress etc.,), and has a test set of 10,000 images.

How it works:
Baseline System

The input, since it is grayscale, is only one channel; each image is 28x28. The network architecture is as follows:
● Layer 1:
  ● 2d convolution of 5x5, 6 feature maps (channels out)
  ● each channel then passes through relu
  ● each channel then passes through max pooling 2x2, stride 2x2
  ● output of layer will be 12x12x6 (12x12 images, 6 channels)
● Layer 2:
  ● 2d convolution of 5x5x6, 12 feature maps out
  ● each channel then passes through relu
  ● each channel then passes through max pooling 2x2, stride 2x2
  ● output of layer will be 4x4x12 (4x4 image maps, 12 channels)
● Layer 3:
  ● fully connected layer, 120 outputs
  ● outputs passed through relu
● Layer 4:
  ● fully connected layer, 60 outputs
  ● outputs passed through relu
● Layer 5:
  ● softmax layer with 10 outputs corresponding to classes

![image](https://user-images.githubusercontent.com/42225976/156088690-e6b8628d-15c3-4899-9f26-33ad3fca8259.png)

