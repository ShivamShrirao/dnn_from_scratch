#!/usr/bin/env python3
import numpy as np

sd=np.random.randint(1000)
# print(sd)
np.random.seed(sd)	#470

def sigmoid(z,a=None,derivative=False,cross=False):
	if derivative:
		return a*(1-a)
	elif cross:
		return 1
	else:
		z=np.clip(z,-500,500)
		return 1.0/(1+np.exp(-z))

def elliot_function(z,a=None, derivative=False,cross=False):
	""" A fast approximation of sigmoid """
	abs_signal=(1+np.abs(z))
	if derivative:
		return 0.5/abs_signal**2
	else:
		return 0.5/abs_signal+0.5

def relu(z,a=None,derivative=False,cross=False):
	if derivative:
		return z>0
	else:
		z[z<0]=0
		return z

def softmax(z,a=None,derivative=False,cross=False):
	if cross:
		return 1
	elif derivative:
		# a*(1-a)
		return 1
	else:
		exps = np.exp(z-np.max(z, axis=1, keepdims = True))
		return exps/np.sum(exps, axis=1, keepdims = True)

def cross_entropy_with_logits(logits,labels):
	return -np.mean(labels*np.log(logits),axis=0,keepdims=True)

def del_cross_sigmoid(pred,labels):
	return (labels-pred)

def del_cross_soft(pred,labels):
	return (labels-pred)

def mean_squared_error(pred, labels):
	return (labels-pred)**2

def del_mean_squared_error(pred, labels):
	return 2*(labels-pred)

def batch_norm(aa):
	gamma=aa.std()
	beta=aa.mean()
	ad=(aa-beta)/gamma				# normalize
	ad=ad*gamma+beta				# recover
	return ad

def echo(z,a=None,derivative=False,cross=False):
	return z