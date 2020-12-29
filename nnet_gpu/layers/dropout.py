#!/usr/bin/env python3
from .base_layer import *


class Dropout(Layer):
	def __init__(
			self,
			rate=0.2,
			name=None
			):		# rate = amount to drop
		saved_locals = locals()		# save for do_init() function
		super().__init__(saved_locals)

	def do_init(self, kwargs):
		input_shape = self.get_inp_shape()
		self.shape = (None, *input_shape)
		self.batches = 1
		self.rate = kwargs.get('rate')
		self.scale = self.dtype(1 / (1 - self.rate))
		self.mask = cp.random.random((self.batches, *input_shape), dtype=self.dtype) > self.rate
		self.param = 0
		self.activation = echo

	def forward(self, inp, training=True):
		if training:		# generate mask with rate probability
			# self.mask = (self.scale * (cp.random.random(inp.shape, dtype=self.dtype) > self.rate)).astype(self.dtype, copy=False)
			self.mask = self.scale * cp.random.binomial(1, 1-self.rate, inp.shape).astype(self.dtype, copy=False)
			return inp * self.mask
		else:
			self.mask.fill(1.0)
			return inp

	def backprop(self, grads, do_d_inp=True):
		return grads * self.mask
