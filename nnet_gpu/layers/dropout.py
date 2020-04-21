#!/usr/bin/env python3
from .base_layer import *
from . import seqinst

class dropout(Layer):
	def __init__(self,rate=0.2,name=None):
		super().__init__()
		self.dtype=cp.float32
		self.type=self.__class__.__name__
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		input_shape=seq_instance.get_inp_shape()
		self.shape=(None,*input_shape)
		self.batches=1
		self.rate=rate
		self.scale=1/(1-rate)
		self.mask=cp.random.random((self.batches,*input_shape))>self.rate
		self.param=0
		self.activation=echo

	def forward(self,inp,training=True):
		if training:
			self.mask=self.scale*cp.random.random(inp.shape)>self.rate 		#generate mask with rate probability
			return inp*self.mask
		else:
			self.mask=inp
			return inp

	def backprop(self,grads,layer=1):
		return grads*self.mask
