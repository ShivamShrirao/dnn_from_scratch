#!/usr/bin/env python3
from ..base_layer import *


class Add(Layer):
	def __init__(
			self,
			name=None
			):
		saved_locals = locals()		# save for do_init() function
		super().__init__(saved_locals)

	def do_init(self, kwargs):
		input_shape = self.get_inp_shape()
		self.shape = (None, *input_shape)
		self.param = 0
		self.activation = echo

	def forward(self, inp, training=True):
		pass

	def backprop(self, grads, do_d_inp=True):
		pass
