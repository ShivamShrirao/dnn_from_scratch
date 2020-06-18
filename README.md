# Deep Learning Library from scratch

A keras like Convolutional Neural Network library made from scratch (just using numpy). Backprop is fully automated. Just specify layers, loss function and optimizers. Model will backpropagate itself.
Just made to learn deep working and backpropogation of CNNs and various machine learning algorithms. Deriving and making it gave alot of insight to how it all works. Will keep adding new networks and algorithms in future.

## Usage

Functions are very much like keras. Check Jupyter notebooks for implementation.

GAN implementation in this library: https://github.com/ShivamShrirao/GANs_from_scratch

### Import modules

```python
from nnet.network import Sequential
from nnet.layers import conv2d,max_pool,flatten,dense,dropout,BatchNormalization
from nnet import optimizers
from nnet import functions
import numpy as np
```

### Make Sequential Model

Add each layer to the Sequential model with parameters.

```python
model = Sequential()

model.add(conv2d(num_kernels=32,kernel_size=3,activation=functions.relu,input_shape=(32,32,3)))
model.add(conv2d(num_kernels=32,kernel_size=3,activation=functions.relu))
model.add(BatchNormalization())
model.add(max_pool())
model.add(dropout(0.1))
model.add(conv2d(num_kernels=64,kernel_size=3,activation=functions.relu))
model.add(conv2d(num_kernels=64,kernel_size=3,activation=functions.relu))
model.add(BatchNormalization())
model.add(max_pool())
model.add(dropout(0.2))
model.add(conv2d(num_kernels=128,kernel_size=3,activation=functions.relu))
model.add(conv2d(num_kernels=128,kernel_size=3,activation=functions.relu))
model.add(BatchNormalization())
model.add(globalAveragePool())
model.add(dropout(0.3))
model.add(dense(10,activation=functions.softmax))
```

### View Model Summary

Shows each layer in a sequence, shape, activations and total, trainable, non-trainable parameters.
TO-DO-> Show connections.

```python
model.summary()
```
```
⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽
Layer (type)               Output Shape             Activation        Param #
==========================================================================================
- input_layer(InputLayer) (None, 32, 32, 3)          echo             0
__________________________________________________________________________________________
0 conv2d(conv2d)          (None, 32, 32, 32)         relu             896
__________________________________________________________________________________________
1 conv2d(conv2d)          (None, 32, 32, 32)         relu             9248
__________________________________________________________________________________________
2 BatchNormalization(Batc (None, 32, 32, 32)         echo             128
__________________________________________________________________________________________
3 max_pool(max_pool)      (None, 16, 16, 32)         echo             0
__________________________________________________________________________________________
4 dropout(dropout)        (None, 16, 16, 32)         echo             0
__________________________________________________________________________________________
5 conv2d(conv2d)          (None, 16, 16, 64)         relu             18496
__________________________________________________________________________________________
6 conv2d(conv2d)          (None, 16, 16, 64)         relu             36928
__________________________________________________________________________________________
7 BatchNormalization(Batc (None, 16, 16, 64)         echo             256
__________________________________________________________________________________________
8 max_pool(max_pool)      (None, 8, 8, 64)           echo             0
__________________________________________________________________________________________
9 dropout(dropout)        (None, 8, 8, 64)           echo             0
__________________________________________________________________________________________
10 conv2d(conv2d)         (None, 8, 8, 128)          relu             73856
__________________________________________________________________________________________
11 conv2d(conv2d)         (None, 8, 8, 128)          relu             147584
__________________________________________________________________________________________
12 BatchNormalization(Bat (None, 8, 8, 128)          echo             512
__________________________________________________________________________________________
13 globalAveragePool(glob (None, 128)                echo             0
__________________________________________________________________________________________
14 dropout(dropout)       (None, 128)                echo             0
__________________________________________________________________________________________
15 dense(dense)           (None, 10)                 softmax          1290
==========================================================================================
Total Params: 289,194
Trainable Params: 288,746
Non-trainable Params: 448
```

### Compile model with optimizer, loss and Learning rate

```python
model.compile(optimizer=optimizers.adam,loss=functions.cross_entropy_with_logits,learning_rate=0.001)
```

### Optimizers avaliable	(nnet.optimizers)

* Iterative			(optimizers.iterative)
* SGD with Momentum (optimizers.momentum)
* Rmsprop			(optimizers.rmsprop)
* Adagrad			(optimizers.adagrad)
* Adam				(optimizers.adam)
* Adamax			(optimizers.adamax)
* Adadelta			(optimizers.adadelta)

### Layers avaliable		(nnet.layers)

* conv2d
* conv2dtranspose
* max_pool
* upsampling
* flatten
* reshape
* dense				(Fully connected layer)
* dropout
* BatchNormalization
* Activation
* InputLayer		(just placeholder)

### Loss functions avaliable	(nnet.functions)

* functions.cross_entropy_with_logits
* functions.mean_squared_error

### Activation Functions avaliable (nnet.functions)

* sigmoid
* elliot
* relu
* leakyRelu
* elu
* tanh
* softmax

### Back Prop

Backprop is fully automated. Just specify layers, loss function and optimizers. Model will backpropagate itself.

### To train

```python
logits=model.fit(X_inp,labels,batch_size=128,epochs=10,validation_data=(X_test,y_test))
```
or
```python
logits=model.fit(X_inp,iterator=img_iterator,batch_size=128,epochs=10,validation_data=(X_test,y_test))
```

### To predict

```python
logits=model.predict(inp)
```

### Save weights and Biases

```python
model.save_weights("file.dump")
```

### Load weights and Biases

```python
model.load_weights("file.dump")
```

## Training graph
Accuracy						 |				Loss
:-------------------------------:|:-------------------------------:
![accuracy](/pics/accuracy.png)	 |	![loss](/pics/loss.png)

## Localization Heatmaps
What the CNN sees
![Heatmap](/pics/localized_heatmap4.png?raw=true)
![Heatmap](/pics/localized_heatmap2.png?raw=true)

## Some predictions.

![automobile](/pics/automobile.png?raw=true)
![deer](/pics/deer.png?raw=true)

## Visualize Feature Maps

![Airplane feature maps](/pics/airplane_feature_maps2.png)

### Digit 6 feature maps
Layer 1

![Number feature maps](/pics/6_feature_maps2.png)

## TO-DO

* RNN and LSTM.
* Write proper tests.
* Auto Differentiation.
* Mixed precision training.
* Multi GPU support. (It still can be done with cupy, just needs proper wrappers)
* Start a server process for visualizing graphs while training.
* Comments.
* Lots of performance and memory improvement.
* Complex architecture like Inception,ResNet.

## References

[CS231n: Convolutional Neural Networks for Visual Recognition.](https://cs231n.github.io/convolutional-networks/)

Original Research Papers. Most implementations are based on original research papers with bit improvements if so.

And a lot of researching on Google.