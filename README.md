# CNN_from_scratch

Convolutional Neural Network module made from scratch. Implemented on cifar and mnist dataset for 
test.
Just a small module for I made to learn deep working and backpropogation of CNNs. Deriving and making it gave alot of insight to how it all works.

## Usage

Functions are very much like tensorflow.

### Import module

```python
import numpy as np
import cnn               # CNN module
import nnet              # ANN module
nn=cnn.conv_net()
```

### Initialize filters and bias

Specify the input channels, kernel size and number of kernels per layer.

```python
w0,b0=nn.init_kernel_bias(num_inp_channels=3,kernel_size=3,num_kernels=64)
w1,b1=nn.init_kernel_bias(num_inp_channels=64,kernel_size=3,num_kernels=128)
w2,b2=nn.init_kernel_bias(num_inp_channels=128,kernel_size=3,num_kernels=128)
w3,b3=nn.init_kernel_bias(num_inp_channels=128,kernel_size=3,num_kernels=256)
```

### Initialize Fully Connected layer

nrons=[Number of inputs, neurons in 1st layer, neurons in 2nd layer, outputs]
Activations in each layer.

```python
import nnet               #ann module
ann=nnet.neural_net(nrons=[4096,256,64,10])
ann.activations(func=['relu','relu','sigmoid'])
```

### Set Learning rate

```python
nn.learning_rate=0.001
ann.learning_rate=0.001
```

### Feed Forward

Specify inputs, weights, activations, pooling, etc. to each layer.

```python
def train(X_inp,y_inp):
    global w0,b0,w1,b1,w2,b2,w3,b3,ann
    # Feed Forward                          #(batches, 32, 32, 3)
    conv0=nn.conv2d(X_inp,w0,b0)
    aconv0=nn.relu(conv0)
#     pool0,max_index0=nn.max_pool(aconv0)  #(batches, 32, 32, 64)

    conv1=nn.conv2d(aconv0,w1,b1)
    aconv1=nn.relu(conv1)
    pool1,max_index1=nn.max_pool(aconv1)    #(batches, 16, 16, 128)

    conv2=nn.conv2d(pool1,w2,b2)
    aconv2=nn.relu(conv2)
    pool2,max_index2=nn.max_pool(aconv2)    #(batches, 8, 8, 128)

    conv3=nn.conv2d(pool2,w3,b3)
    aconv3=nn.relu(conv3)
    pool3,max_index3=nn.max_pool(aconv3)    #(batches, 4, 4, 256)

    r,c,d=pool3.shape[1:]
    flat=pool3.reshape(-1,r*c*d)

    err3=np.empty(flat.shape)
    for i,flat_layer in enumerate(flat):
        ott=ann.feed_forward(flat_layer)
      # prediction=out.argmax()
```
### Back Prop
```python
        err3[i]=ann.backprop(y_inp[i])[0]
    err3=err3.reshape(-1,r,c,d)            #(batches, 4, 4, 256)
    err3=np.array(err3).reshape(-1,r,c,d)  #(batches, 4, 4, 256)
    # Back prop CNN
    d_aconv3=nn.max_pool_back(errors=err3,inp=aconv3,max_index=max_index3)
    d_conv3=d_aconv3*nn.relu_der(aconv3,conv3)
    d_pool2,d_w3,d_b3=nn.conv2d_back(errors=d_conv3,inp=pool2,kernels=w3,biases=b3)
    w3+=d_w3
    b3+=d_b3

    d_aconv2=nn.max_pool_back(errors=d_pool2,inp=aconv2,max_index=max_index2)
    d_conv2=d_aconv2*nn.relu_der(aconv2,conv2)
    d_pool1,d_w2,d_b2=nn.conv2d_back(errors=d_conv2,inp=pool1,kernels=w2,biases=b2)
    w2+=d_w2
    b2+=d_b2

    d_aconv1=nn.max_pool_back(errors=d_pool1,inp=aconv1,max_index=max_index1)
    d_conv1=d_aconv1*nn.relu_der(aconv1,conv1)
    d_aconv0,d_w1,d_b1=nn.conv2d_back(errors=d_conv1,inp=aconv0,kernels=w1,biases=b1)
    w1+=d_w1
    b1+=d_b1

#     d_aconv0=nn.max_pool_back(errors=d_pool0,inp=aconv0,max_index=max_index0)
    d_conv0=d_aconv0*nn.relu_der(aconv0,conv0)
    d_inp,d_w0,d_b0=nn.conv2d_back(errors=d_conv0,inp=X_inp,kernels=w0,biases=b0,layer=0) # layer 0 specifies to not back prop X_inp
    w0+=d_w0
    b0+=d_b0
```
#### *errors here specify the gradient*

## Some predictions.

![automobile](/pics/automobile.png?raw=true)
![dog](/pics/dog.png?raw=true)

## Visualize Feature Maps

![Airplane](/pics/airplane.png?raw=true)
![Airplane feature maps](/pics/airplane_feature_maps.png?raw=true)
![Airplane feature maps](/pics/airplane_feature_maps2.png?raw=true)

### Digit 6 feature maps
Layer 0

![Number](/pics/6_feature_maps1.png?raw=true)

Layer 1

![Number feature maps](/pics/6_feature_maps2.png?raw=true)

Activations

![Number feature maps](/pics/6_feature_maps3.png?raw=true)

## TO-DO

* Automate Backprop sequencing.
* Add batches to fully connected layer.(ANN)
* Extend current im2col to batches to eliminate for loop and shift all calculations to numpy.
* Add more helper functions.
* Batch normalization.
* Add better loss calculator and optimizer.
* Lots of performance and memory improvement.

## References

[CS231n: Convolutional Neural Networks for Visual Recognition.](https://cs231n.github.io/convolutional-networks/)

And a lot of researching on Google.