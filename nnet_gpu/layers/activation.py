#!/usr/bin/env python3
from .base_layer import *
from ..stream_handler import stream_maps


class Activation(Layer):
	def __init__(
			self,
			activation=echo,
			input_shape=None,
			name=None
			):
		saved_locals = locals()		# save for do_init() function
		super().__init__(saved_locals)

	def do_init(self, kwargs):
		self.input_shape = kwargs.get('input_shape')
		if self.input_shape is None:
			self.input_shape = self.get_inp_shape()
		self.activation = kwargs.get('activation')
		self.shape = (None, *self.input_shape)
		self.param = 0
		self.not_softmax_cross_entrp = True
		if self.activation == echo:
			self.notEcho = False
		else:
			self.notEcho = True

	def forward(self, inp, training=True):
		self.z_out = inp
		self.a_out = self.activation(self.z_out)
		return self.a_out

	def backprop(self, grads, do_d_inp=True):
		if self.notEcho and self.not_softmax_cross_entrp:		# TODO: Use proper checks.
			grads *= self.activation(self.z_out, self.a_out, derivative=True)
		return grads
