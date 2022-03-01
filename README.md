# Convolutional-Neural-Network-for-Fashion_MNIST-
Implementation of CNN for Fashion-MNIST dataset

Fashion-MNIST dataset consists of 60,000 28x28 grayscale training images of clothing items of one of ten types (shirt, dress etc.,), and has a test set of 10,000 images.

How it works:
Baseline System

The input, since it is grayscale, is only one channel; each image is 28x28. The network architecture is as follows:

Layer 1:

  ● 2d convolution of 5x5, 6 feature maps (channels out)
  
  ● each channel then passes through relu
  
  ● each channel then passes through max pooling 2x2, stride 2x2
  
  ● output of layer will be 12x12x6 (12x12 images, 6 channels)
  
Layer 2:

  ● 2d convolution of 5x5x6, 12 feature maps out
  
  ● each channel then passes through relu
  
  ● each channel then passes through max pooling 2x2, stride 2x2
  
  ● output of layer will be 4x4x12 (4x4 image maps, 12 channels)
  
Layer 3:

  ● fully connected layer, 120 outputs
  
  ● outputs passed through relu
  
Layer 4:

  ● fully connected layer, 60 outputs
  
  ● outputs passed through relu
  
Layer 5:

  ● softmax layer with 10 outputs corresponding to classes

![image](https://user-images.githubusercontent.com/42225976/156088690-e6b8628d-15c3-4899-9f26-33ad3fca8259.png)

After implementing the baseline model in Pytorch, the results obtained for learning rate of 0.001, batch size of 1000 and for 10 epochs are,

![image](https://user-images.githubusercontent.com/42225976/156088984-6d945841-bd8b-45f7-8c7a-59ab4a09e202.png)

Comparison Systems
Different variants of the systems are implemented to check the performance on the test set.

i) System with different number of layers

Changes made to the baseline model include, having only one fully connected layer while keeping everything else in the baseline model intact. The results of the implementation of this model are,

![image](https://user-images.githubusercontent.com/42225976/156089076-abcbddd0-13df-45c4-a0b4-f16369dd2e7f.png)

ii) System with different layer sizes

Changes made to the baseline model include, changing the kernel size from (5,5) to (3,3) which results in a different size of the convolutional layer and input to the first fully connected layer. Everything else is the same as that of the baseline model. The results of the implementation of this model are,

![image](https://user-images.githubusercontent.com/42225976/156089142-88f87065-b69f-442a-8b07-9927e6c76aa1.png)

iii) System with different number of feature maps

Changes made to the baseline model include, the output channels/feature maps of the first convolutional layer are 32 and the output channels/feature maps of the second convolutional layer are 64 while keeping everything else in the baseline model intact. The results of the implementation of this model are,

![image](https://user-images.githubusercontent.com/42225976/156089204-07d516e2-9b9c-46af-9974-b208934eb7eb.png)

iv) System with different layer sizes, different number of feature maps and different kernel size

Changes made to the baseline model include, the output channels of the first convolution layer are 32 and output channels of the second convolution layer are 64 and the kernel size being changed from (5,5) to (3,3) and having only one fully connected layer and a dropout of 25% of the nodes implemented after fully connected layer 1 while keeping everything else in the baseline model intact. The results of the implementation of this model are,

![image](https://user-images.githubusercontent.com/42225976/156089262-d74ac293-4674-4fd5-9ca3-f14c36f5d04f.png)

v) Changing pooling attributes:

Changing the pooling attribute from max pooling to average pooling is not good for object detection or edge detection type tasks because although average pooling extracts features in a smooth way, it can’t extract the good features effectively as opposed to max pooling which is better for extracting edges. Hence, max pooling for most cases provides better test accuracy than average pooling. Since the implementation of average pooling reduced the test accuracy compared to the baseline model for this dataset.
