# Deep Learning Library from scratch

A keras like Convolutional Neural Network library made from scratch (just using numpy). Backprop is fully automated. Just specify layers, loss function and optimizers. Model will backpropagate itself.
Just made to learn deep working and backpropogation of CNNs and various machine learning algorithms. Deriving and making it gave alot of insight to how it all works. Will keep adding new networks and algorithms in future.

## Usage

Functions are very much like keras.

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
model.add(conv2d(num_kernels=32,kernel_size=3,activation=functions.relu,input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(max_pool())
model.add(dropout(0.1))
model.add(conv2d(num_kernels=64,kernel_size=3,activation=functions.relu))
model.add(BatchNormalization())
model.add(max_pool())
model.add(dropout(0.2))
model.add(conv2d(num_kernels=128,kernel_size=3,activation=functions.relu))
model.add(BatchNormalization())
model.add(max_pool())
model.add(dropout(0.3))
model.add(flatten())
model.add(dense(512,activation=functions.relu))
model.add(BatchNormalization())
model.add(dropout(0.4))
model.add(dense(10,activation=functions.softmax))
```

### View Model Summary

Shows each layer in a sequence, shape, activations and total, trainable, non-trainable parameters.
TO-DO-> Show connetions.

```python
model.summary()
```
```
⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽
Layer (type)               Output Shape             Activation        Param #
==========================================================================================
input_layer (InputLayer)  (None, 32, 32, 3)          echo             0
__________________________________________________________________________________________
conv2d (conv2d)           (None, 32, 32, 32)         relu             896
__________________________________________________________________________________________
BatchNormalization (Batch (None, 32, 32, 32)         echo             128
__________________________________________________________________________________________
max_pool (max_pool)       (None, 16, 16, 32)         echo             0
__________________________________________________________________________________________
dropout (dropout)         (None, 16, 16, 32)         echo             0
__________________________________________________________________________________________
conv2d (conv2d)           (None, 16, 16, 64)         relu             18496
__________________________________________________________________________________________
BatchNormalization (Batch (None, 16, 16, 64)         echo             256
__________________________________________________________________________________________
max_pool (max_pool)       (None, 8, 8, 64)           echo             0
__________________________________________________________________________________________
dropout (dropout)         (None, 8, 8, 64)           echo             0
__________________________________________________________________________________________
conv2d (conv2d)           (None, 8, 8, 128)          relu             73856
__________________________________________________________________________________________
BatchNormalization (Batch (None, 8, 8, 128)          echo             512
__________________________________________________________________________________________
max_pool (max_pool)       (None, 4, 4, 128)          echo             0
__________________________________________________________________________________________
dropout (dropout)         (None, 4, 4, 128)          echo             0
__________________________________________________________________________________________
flatten (flatten)         (None, 2048)               echo             0
__________________________________________________________________________________________
dense (dense)             (None, 512)                relu             1049088
__________________________________________________________________________________________
BatchNormalization (Batch (None, 512)                echo             2048
__________________________________________________________________________________________
dropout (dropout)         (None, 512)                echo             0
__________________________________________________________________________________________
dense (dense)             (None, 10)                 softmax          5130
==========================================================================================
Total Params: 1,150,410
Trainable Params: 1,148,938
Non-trainable Params: 1,472
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
* max_pool
* flatten
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
* elu
* tanh
* softmax

### To train

```python
logits=model.fit(inp,y_inp)
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

### Back Prop

Backprop is fully automated. Just specify layers, loss function and optimizers. Model will backpropagate itself.

## Training graph
### Accuracy
![accuracy](/pics/accuracy.png?raw=true)
### Loss
![loss](/pics/loss.png?raw=true)

## Some predictions.

![automobile](/pics/automobile.png?raw=true)
![deer](/pics/deer.png?raw=true)
![dog](/pics/dog.png?raw=true)
![horse](/pics/horse.png?raw=true)

## Visualize Feature Maps

![Airplane](/pics/airplane.png?raw=true)
![Airplane feature maps](/pics/airplane_feature_maps.png?raw=true)
![Airplane feature maps](/pics/airplane_feature_maps2.png?raw=true)

### Digit 6 feature maps
Layer 1

![Number feature maps](/pics/6_feature_maps2.png?raw=true)

Activations

![Number feature maps](/pics/6_feature_maps3.png?raw=true)

## TO-DO

* Start a server process for visualizing graphs while training.
* L2 norm Regulization.
* Lots of performance and memory improvement.
* Complex architecture like ResNet ?
* GPU support.

## References

[CS231n: Convolutional Neural Networks for Visual Recognition.](https://cs231n.github.io/convolutional-networks/)

Original Research Papers. Most implementations are based on original research papers with bit improvements if so.

And a lot of researching on Google.