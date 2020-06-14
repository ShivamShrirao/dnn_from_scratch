#!/usr/bin/env python3
from .base_layer import *


class dropout(Layer):
	def __init__(self, rate=0.2, name=None):  # rate = amount to drop
		super().__init__()
		self.dtype = cp.float32
		self.type = self.__class__.__name__
		if name is None:
			self.name = self.__class__.__name__
		else:
			self.name = name
		input_shape = self.get_inp_shape()
		self.shape = (None, *input_shape)
		self.batches = 1
		self.rate = rate
		self.scale = cp.float32(1 / (1 - rate))
		self.mask = cp.random.random((self.batches, *input_shape), dtype=cp.float32) > self.rate
		self.param = 0
		self.activation = echo

	def forward(self, inp, training=True):
		if training:		# generate mask with rate probability
			self.mask = (self.scale * (cp.random.random(inp.shape, dtype=cp.float32) > self.rate)).astype(cp.float32, copy=False)
			return inp * self.mask
		else:
			self.mask.fill(1.0)
			return inp

	def backprop(self, grads, layer=1):
		return grads * self.mask
