#!/usr/bin/env python3
import numpy as np

sd=np.random.randint(1000)
print("Seed:",sd)
np.random.seed(sd)

### CAN TURN THESE INTO CLASSES

def sigmoid(z,a=None,derivative=False):
	if derivative:
		return a*(1-a)
	else:
		z=z.clip(-500,500)
		return 1.0/(1+np.exp(-z))

def elliot(z,a=None, derivative=False):
	# A fast approximation of sigmoid
	abs_signal=(1+np.abs(z))
	if derivative:
		return 0.5/abs_signal**2
	else:
		return 0.5/abs_signal+0.5

def relu(z,a=None,derivative=False):
	if derivative:
		return z>0
	else:
		z[z<0]=0
		return z

def elu(z,a=None,derivative=False):			#alpha is 1
	if derivative:
		return np.where(z>0, 1, a+1)
	else:
		return np.where(z>0, z, np.exp(z)-1)

def leakyRelu(z,a=None,derivative=False):		#alpha is 0.01
	alpha=0.01
	if derivative:
		dz = np.ones_like(z)
		dz[z < 0] = alpha
		return dz
	else:
		return np.where(z>0, z, z*alpha)

def tanh(z,a=None,derivative=False):
	if derivative:
		return 1-a**2
	else:
		return np.tanh(z)

def softmax(z,a=None,derivative=False):
	if derivative:
		# a1*(1-a1)-a1a2
		return 1
	else:
		exps = np.exp(z-np.max(z, axis=1, keepdims = True))
		return exps/np.sum(exps, axis=1, keepdims = True)

def cross_entropy_with_logits(logits,labels):
	return -np.mean(labels*np.log(logits+1e-30),axis=0,keepdims=True)

def del_cross_sigmoid(logits,labels):
	return (logits-labels)

def del_cross_soft(logits,labels):
	return (logits-labels)

def mean_squared_error(logits, labels):
	return ((logits-labels)**2)/2

def del_mean_squared_error(logits, labels):
	return (logits-labels)

def echo(z,a=None,derivative=False):
	return z