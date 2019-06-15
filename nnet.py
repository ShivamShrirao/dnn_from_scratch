#!/usr/bin/env python3
import numpy as np

sd=470#np.random.randint(1000)
# print(sd)
np.random.seed(sd)	#470

class neural_net:
	def __init__(self):
		self.learning_rate=0.01

	def __str__(self):
		return str(self.__dict__)

	def init_weights_bias(self,num_inp,num_out,mean=0,std=0.1):
		weights = std*np.random.randn(num_inp,num_out) + mean
		bias = std*np.random.randn(1,num_out) + mean
		return weights,bias

	def sigmoid(self,x):
		x=np.clip(x,-500,500)
		return 1.0/(1+np.exp(-x))

	def sigmoid_der(self,y):
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

	def relu(self,x):
		x[x<0]=0
		return x

	def relu_der(self,x,y):
		return (y > 0)

	def softmax(self,x):
		# exps = np.exp(x)
		exps = np.exp(x-np.max(x, axis=1, keepdims = True))
		return exps/np.sum(exps, axis=1, keepdims = True)

	def soft_der(self,x,y):
		return np.ones(self.softmax(x).shape)

	def del_cross_soft(self,out,res):
		res = res.argmax(axis=1)
		m = res.shape[0]
		grad = out
		grad[range(m),res]-=1
		grad = grad/m
		return grad

	def batch_norm(self,aa):
		gamma=aa.std()
		beta=aa.mean()
		ad=(aa-beta)/gamma				# normalize
		ad=ad*gamma+beta				# recover
		return ad

	def mean_squared_error(self, pred, labels):
		return (labels-pred)**2

	def mean_squared_error_der(self, pred, labels):
		return 2*(labels-pred)

	def feed_forward(self, X, weights, biases):
		return np.dot(X,weights)+biases

	def backprop(self,errors,inp,weights,biases,layer=1):
		d_c_b=errors
		d_c_w=np.dot(inp.T,d_c_b)/inp.shape[0]
		if layer:
			d_c_a=np.dot(d_c_b,weights.T)
		else:
			d_c_a=0
		return d_c_a, d_c_w*self.learning_rate, d_c_b.mean(axis=0,keepdims=True)*self.learning_rate