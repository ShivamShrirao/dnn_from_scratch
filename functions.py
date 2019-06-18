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

def del_cross_sigmoid(logits,labels):
	return (logits-labels)

def del_cross_soft(logits,labels):
	return (logits-labels)

def mean_squared_error(logits, labels):
	return ((logits-labels)**2)/2

def del_mean_squared_error(logits, labels):
	return (logits-labels)

def batch_norm(aa):
	gamma=aa.std()
	beta=aa.mean()
	ad=(aa-beta)/gamma				# normalize
	ad=ad*gamma+beta				# recover
	return ad

def echo(z,a=None,derivative=False,cross=False):
	return z

def iterative(sequence,learning_rate=0.01):
	for obj in sequence:
		if obj.param>0:
			obj.kernels-=obj.d_c_w*learning_rate
			obj.biases-=obj.d_c_b*learning_rate

def momentum(sequence,learning_rate=0.01,beta=0.9):
	for obj in sequence:
		if obj.param>0:
			obj.w_m=beta*obj.w_m - learning_rate*obj.d_c_w
			obj.kernels+=obj.w_m
			obj.b_m=beta*obj.b_m - learning_rate*obj.d_c_b
			obj.biases+=obj.b_m

def adam(sequence,learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8,decay=0):
	for obj in sequence:
		if obj.param>0:
			# Update weights
			obj.w_m=beta1*obj.w_m + (1-beta1)*obj.d_c_w
			obj.w_v=beta1*obj.w_v + (1-beta2)*(obj.d_c_w**2)
			mcap=obj.w_m/(1-beta1)
			vcap=obj.w_v/(1-beta2)
			obj.d_c_w=mcap/(np.sqrt(vcap)+epsilon)
			obj.kernels-=obj.d_c_w*learning_rate
			# Update biases
			obj.b_m=beta1*obj.b_m + (1-beta1)*obj.d_c_b
			obj.b_v=beta1*obj.b_v + (1-beta2)*(obj.d_c_b**2)
			mcap=obj.b_m/(1-beta1)
			vcap=obj.b_v/(1-beta2)
			obj.d_c_b=mcap/(np.sqrt(vcap)+epsilon)
			obj.biases-=obj.d_c_b*learning_rate