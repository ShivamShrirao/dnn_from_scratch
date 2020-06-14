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
		super().__init__()
		self.dtype = cp.float32
		self.type = self.__class__.__name__
		if name is None:
			self.name = self.__class__.__name__
		else:
			self.name = name
		if input_shape is None:
			input_shape = self.get_inp_shape()
		self.activation = activation
		self.shape = (None, *input_shape)
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

	def backprop(self, grads, layer=1):
		if self.notEcho and self.not_softmax_cross_entrp:
			grads *= self.activation(self.z_out, self.a_out, derivative=True)
		return grads
