#!/usr/bin/env python3
import numpy as np

sd=np.random.randint(1000)
# print(sd)
np.random.seed(sd)	#470

def sigmoid(x):
	x=np.clip(x,-500,500)
	return 1.0/(1+np.exp(-x))

def sigmoid_der(y):
	return y * (1 - y)

def elliot_function(signal, derivative=False):
	""" A fast approximation of sigmoid """
	s = 1 # steepness
	
	abs_signal = (1 + np.abs(signal * s))
	if derivative:
		return 0.5 * s / abs_signal**2
	else:
		# Return the activation signal
		return 0.5*(signal * s) / abs_signal + 0.5

def relu(x):
	x[x<0]=0
	return x

def relu_der(x,y):
	return (y > 0)

def softmax(x):
	# exps = np.exp(x)
	exps = np.exp(x-np.max(x, axis=1, keepdims = True))
	return exps/np.sum(exps, axis=1, keepdims = True)

def soft_der(x,y):
	return np.ones(softmax(x).shape)

def cross_entropy_with_logits(logits,labels):
	return -np.mean(labels*np.log(logits),axis=0,keepdims=True)

def del_cross_sigmoid(pred,labels):
	return (labels-pred)

def del_cross_soft(pred,labels):
	return (labels-pred)

def mean_squared_error( pred, labels):
	return (labels-pred)**2

def mean_squared_error_der( pred, labels):
	return 2*(labels-pred)

def batch_norm(aa):
	gamma=aa.std()
	beta=aa.mean()
	ad=(aa-beta)/gamma				# normalize
	ad=ad*gamma+beta				# recover
	return ad