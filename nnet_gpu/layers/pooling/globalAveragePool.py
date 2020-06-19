#!/usr/bin/env python3
from .maxpool import *


class GlobalAveragePool(Layer):
	def __init__(
			self,
			input_shape=None,
			name=None
			):
		saved_locals = locals()		# save for do_init() function
		super().__init__(saved_locals)

	def do_init(self, kwargs):
		self.input_shape = kwargs.get('input_shape')
		if self.input_shape is None:
			self.input_shape = self.get_inp_shape()
		self.param = 0
		self.batches = 1
		self.row, self.col, self.channels = self.input_shape
		self.Ncount = self.row * self.col
		self.shape = (None, self.channels)

	def forward(self, inp, training=True):
		self.input_shape = inp.shape
		self.batches = self.input_shape[0]
		inp = inp.Reshape(self.batches, self.Ncount, self.channels)
		output = inp.mean(axis=1)
		return output.Reshape(self.batches, self.channels)

	def backprop(self, grads, do_d_inp=True):
		# grads/=self.Ncount
		z_out = grads.repeat(self.Ncount, axis=0).Reshape(self.batches, self.row, self.col, self.channels)
		return z_out
