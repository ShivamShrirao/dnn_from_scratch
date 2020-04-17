#!/usr/bin/env python3
from nnet_gpu.functions import *
import numpy as np
import cupy as cp

sd=np.random.randint(1000)
print("Seed:",sd)
np.random.seed(sd)
cp.random.seed(sd)

class Layer:
	def __init__(self):
		self.name=self.__class__.__name__
		self.type=self.__class__.__name__
		self.dtype=np.float32
		self.param=0
		self.activation=echo
		self.input_layer=None
		self.output_layers=[]

	def __str__(self):
		return self.name+super().__str__()

	def __call__(self,lyr):
		self.input_layer=lyr
		lyr.output_layers.append(self)
		return self

class conv2d(Layer):
	def __init__(self):
		super().__init__()
		

class dense(Layer):
	def __init__(self,num_out,input_shape=None,weights=None,biases=None,activation=echo,mean=0,std=0.01,name=None):
		super().__init__()
		self.dtype=np.float32
		self.type=self.__class__.__name__
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		if input_shape is None:
			self.input_shape=seq_instance.get_inp_shape()[0]
		else:
			self.input_shape=input_shape
		self.activation=activation
		if weights is None:
			self.weights = std*cp.random.randn(self.input_shape,num_out).astype(self.dtype,copy=False) + mean
			# weights/=np.sqrt(self.input_shape)
		else:
			if weights.shape!=(self.input_shape,num_out):
				raise Exception("weights should be of shape: "+str((self.input_shape,num_out)))
			else:
				self.weights = cp.asarray(weights)
		if biases is None:
			self.biases = std*cp.random.randn(1,num_out).astype(self.dtype,copy=False) + mean
		else:
			if biases.shape!=(1,num_out):
				raise Exception("biases should be of shape: "+str((1,num_out)))
			else:
				self.biases = cp.asarray(biases)
		self.kernels = self.weights
		self.w_m=cp.zeros_like(self.weights)
		self.w_v=cp.zeros_like(self.weights)
		self.b_m=cp.zeros_like(self.biases)
		self.b_v=cp.zeros_like(self.biases)
		self.shape=(None,num_out)
		self.param=self.input_shape*num_out + num_out
		self.not_softmax_cross_entrp=True
		if self.activation==echo:
			self.notEcho=False
		else:
			self.notEcho=True

	def forward(self,inp,training=True):
		self.inp=cp.asarray(inp)
		self.z_out=self.inp.dot(self.weights)+self.biases
		self.a_out=self.activation(self.z_out)
		return self.a_out

	def backprop(self,grads,layer=1):
		if self.notEcho and self.not_softmax_cross_entrp:			# make it better in future
			grads*=self.activation(self.z_out,self.a_out,derivative=True)
		d_c_b=grads
		self.d_c_w=self.inp.T.dot(d_c_b)#/self.inp.shape[0]
		if layer:
			d_c_a=d_c_b.dot(self.weights.T)
		else:
			d_c_a=0
		self.d_c_b=d_c_b.sum(axis=0,keepdims=True)
		# self.d_c_b=d_c_b.mean(axis=0,keepdims=True)
		return d_c_a

class BatchNormalization(Layer):				# NOT IMPLEMENTED
	def __init__(self,momentum=0.9,epsilon=1e-10,name=None):
		super().__init__()
		self.dtype=np.float32
		self.type=self.__class__.__name__
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		input_shape=seq_instance.get_inp_shape()
		self.shape=(None,*input_shape)
		self.batches=1
		self.inp_shape=(self.batches,*input_shape)
		self.biases=np.zeros(input_shape).astype(self.dtype)		#biases is beta
		self.weights=np.ones(input_shape).astype(self.dtype)		#weights is gamma
		self.gamma=self.weights
		self.beta=self.biases
		self.kernels = self.weights
		self.w_m=cp.zeros_like(self.weights)
		self.w_v=cp.zeros_like(self.weights)
		self.b_m=cp.zeros_like(self.biases)
		self.b_v=cp.zeros_like(self.biases)
		self.epsilon=epsilon
		self.momentum=momentum
		self.moving_mean=None
		self.moving_var=None
		self.param=4*input_shape[-1]
		self.activation=echo

	def forward(self,inp,training=True):		# yeah, I know, too many repetitions
		#inp[batches,row,col,channels]
		if training:
			self.inp_shape=inp.shape
			self.mean=inp.mean(axis=0)					#(row,col,channels)
			self.xmu=inp-self.mean 						#(batches,row,col,channels)
			self.var=(self.xmu**2).mean(axis=0)			#(row,col,channels)
			self.ivar=1/(self.var+self.epsilon)			#(row,col,channels)
			self.istd=np.sqrt(self.ivar)				#(row,col,channels)
			self.xnorm=self.xmu*self.istd 				#(batches,row,col,channels)
			if self.moving_mean is None:
				self.moving_mean=self.mean
				self.moving_var=self.var
			else:
				self.moving_mean=self.momentum*self.moving_mean + (1-self.momentum)*self.mean
				self.moving_var=self.momentum*self.moving_var + (1-self.momentum)*self.var
		else:
			if self.moving_mean is None:
				self.inp_shape=inp.shape
				self.mean=inp.mean(axis=0)					#(row,col,channels)
				self.xmu=inp-self.mean 						#(batches,row,col,channels)
				self.var=(self.xmu**2).mean(axis=0)			#(row,col,channels)
				self.ivar=1/(self.var+self.epsilon)			#(row,col,channels)
				self.istd=np.sqrt(self.ivar)				#(row,col,channels)
				self.moving_mean=self.mean
				self.moving_var=self.var
				self.xnorm=self.xmu*self.istd 				#(batches,row,col,channels)
			else:
				self.inp_shape=inp.shape
				self.xmu=inp								#(batches,row,col,channels)	## all this is just for proper shape while model.free()
				self.istd=self.moving_var					#(row,col,channels)
				self.xnorm=(inp-self.moving_mean)/np.sqrt(self.moving_var+self.epsilon)
		return self.xnorm*self.weights+self.biases

	def backprop(self,grads,layer=1):
		#grads(batches,row,col,channels), xmu(batches,row,col,channels)=inp-mean 		#FU
		batches=self.inp_shape[0]
		if batches!=self.batches:
			self.batches=batches
		self.d_c_b=grads.sum(axis=0) 				#(row,col,channels)		# biases is beta
		self.d_c_w=(self.xnorm*grads).sum(axis=0)	#(row,col,channels)		# gamma is weights
		d_inp=(1/self.batches)*self.istd*self.weights*(self.batches*grads-self.d_c_b-self.xmu*self.ivar*((grads*self.xmu).sum(axis=0)))
		return d_inp


class InputLayer(Layer):		#just placeholder
	def __init__(self,shape=None):
		super().__init__()
		self.name='input_layer'
		self.type=self.__class__.__name__
		self.dtype=np.float32
		try:
			self.shape=(None,*shape)
		except:
			self.shape=(None,shape)
		self.param=0
		self.activation=echo