#!/usr/bin/env python3
from ..functions import *
import numpy as np
import cupy as cp

class Layer:
	def __init__(self):
		self.name=self.__class__.__name__
		self.type=self.__class__.__name__
		self.dtype=np.float32
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