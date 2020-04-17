#!/usr/bin/env python3
import numpy as np
import cupy as cp

### CAN TURN THESE INTO CLASSES

def sigmoid(z,a=None,derivative=False):
	if derivative:
		return a*(1-a)
	else:
		return 1.0/(1+cp.exp(-z.clip(-88.72283,88.72283)))

def elliot(z,a=None, derivative=False):
	# A fast approximation of sigmoid
	abs_signal=(1+cp.abs(z))
	if derivative:
		return 0.5/abs_signal**2
	else:
		return 0.5/abs_signal+0.5

def relu(z,a=None,derivative=False):
	if derivative:
		return z>0
	else:
		# z[z<0]=0
		# return z
		# return z*(z>0)
		return cp.maximum(0,z)

def elu(z,a=None,derivative=False):			#alpha is 1
	if derivative:
		return cp.where(z>0, 1, a+1)
	else:
		return cp.where(z>0, z, cp.exp(z)-1)#*alpha

def leakyRelu(z,a=None,derivative=False):
	alpha=0.2
	if derivative:
		# dz = cp.ones_like(z,dtype=cp.float32)
		# dz[z < 0] = alpha
		# return dz
		return cp.clip(z > 0, alpha, 1.0)
	else:
		return cp.where(z>0, z, z*alpha)

def tanh(z,a=None,derivative=False):
	if derivative:
		return 1-a**2
	else:
		return cp.tanh(z)

def softmax(z,a=None,derivative=False):
	if derivative:
		# a1*(1-a1)-a1a2
		return 1
	else:
		exps = cp.exp(z-cp.max(z, axis=1, keepdims = True))
		return exps/cp.sum(exps, axis=1, keepdims = True)

def cross_entropy_with_logits(logits,labels,epsilon=1e-12):
	return -cp.sum(labels*cp.log(logits+epsilon),axis=0,keepdims=True)

def cross_entropy(logits,labels,epsilon=1e-12):
	labels=labels.clip(epsilon,1-epsilon)
	logits=logits.clip(epsilon,1-epsilon)
	return -labels*cp.log(logits)-(1-labels)*cp.log(1-logits)

def del_cross_sigmoid(logits,labels):
	return (logits-labels)

def del_cross_soft(logits,labels):
	return (logits-labels)

def mean_squared_error(logits, labels):
	return ((logits-labels)**2)/2

def del_mean_squared_error(logits, labels):
	return (logits-labels)

def echo(z,a=None,derivative=False,**kwargs):
	return z