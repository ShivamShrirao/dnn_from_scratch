#!/usr/bin/env python3
from .base_layer import *
from . import seqinst


class flatten(Layer):
	def __init__(self, name=None):
		super().__init__()
		self.type = self.__class__.__name__
		self.dtype = cp.float32
		if name is None:
			self.name = self.__class__.__name__
		else:
			self.name = name
		input_shape = seqinst.seq_instance.get_inp_shape()
		self.r, self.c, self.channels = input_shape
		self.fsz = self.r * self.c * self.channels
		self.shape = (None, self.fsz)
		self.param = 0
		self.activation = echo

	def forward(self, inp, training=True):
		return inp.reshape(-1, self.fsz)

	def backprop(self, grads, layer=1):
		return grads.reshape(-1, self.r, self.c, self.channels)


class reshape(Layer):
	def __init__(self, target_shape, name=None):
		super().__init__()
		self.type = self.__class__.__name__
		self.dtype = cp.float32
		if name is None:
			self.name = self.__class__.__name__
		else:
			self.name = name
		self.input_shape = seqinst.seq_instance.get_inp_shape()
		self.target_shape = target_shape
		tt = 1
		for i in self.input_shape:
			tt *= i
		for i in target_shape:
			tt /= i
		if tt != 1:
			raise Exception("Cannot reshape input " + str(self.input_shape) + " to " + str(target_shape) + '.')
		self.shape = (None, *target_shape)
		self.param = 0
		self.activation = echo

	def forward(self, inp, training=True):
		return inp.reshape(-1, *self.target_shape)

	def backprop(self, grads, layer=1):
		return grads.reshape(-1, *self.input_shape)
