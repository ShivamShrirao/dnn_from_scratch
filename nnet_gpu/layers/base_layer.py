#!/usr/bin/env python3
from ..functions import *
import numpy as np
import cupy as cp
from . import seqinst

class Layer:
	def __init__(self):
		self.name=self.__class__.__name__
		self.type=self.__class__.__name__
		self.dtype=cp.float32
		self.param=0
		self.activation=echo
		self.input_layer=None
		self.output_layers=[]
		self.bias_is_not_0=True

	def __str__(self):
		return self.name+super().__str__()

	def __call__(self,lyr):
		self.input_layer=lyr
		lyr.output_layers.append(self)
		return self

class Activation(Layer):
	def __init__(self,activation=echo,input_shape=None,name=None):
		super().__init__()
		self.dtype=cp.float32
		self.type=self.__class__.__name__
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		if input_shape is None:
			input_shape=seqinst.seq_instance.get_inp_shape()
		self.activation=activation
		self.shape=(None,*input_shape)
		self.param=0
		self.not_softmax_cross_entrp=True
		if self.activation==echo:
			self.notEcho=False
		else:
			self.notEcho=True

	def forward(self,inp,training=True):
		self.z_out=inp
		self.a_out=self.activation(self.z_out)
		return self.a_out

	def backprop(self,grads,layer=1):
		if self.notEcho and self.not_softmax_cross_entrp:
			grads*=self.activation(self.z_out,self.a_out,derivative=True)
		return grads

class InputLayer(Layer):		#just placeholder
	def __init__(self,shape=None):
		super().__init__()
		self.name='input_layer'
		self.type=self.__class__.__name__
		self.dtype=cp.float32
		try:
			self.shape=(None,*shape)
		except:
			self.shape=(None,shape)
		self.param=0
		self.activation=echo