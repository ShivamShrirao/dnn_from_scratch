#!/usr/bin/env python3
from .max_pool import *


class globalAveragePool(Layer):
	def __init__(
			self,
			input_shape=None,
			name=None
			):
		super().__init__()
		self.type = self.__class__.__name__
		if name is None:
			self.name = self.__class__.__name__
		else:
			self.name = name
		if input_shape is None:
			input_shape = self.get_inp_shape()
		self.param = 0
		self.batches = 1
		self.row, self.col, self.channels = input_shape
		self.Ncount = self.row * self.col
		self.shape = (None, self.channels)

	def forward(self, inp, training=True):
		self.input_shape = inp.shape
		self.batches = self.input_shape[0]
		inp = inp.reshape(self.batches, self.Ncount, self.channels)
		output = inp.mean(axis=1)
		return output.reshape(self.batches, self.channels)

	def backprop(self, grads, layer=1):
		# grads/=self.Ncount
		z_out = grads.repeat(self.Ncount, axis=0).reshape(self.batches, self.row, self.col, self.channels)
		return z_out
