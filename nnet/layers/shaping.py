#!/usr/bin/env python3
from .base_layer import *


class Flatten(Layer):
	def __init__(
			self,
			name=None
			):
		saved_locals = locals()  # save for do_init() function
		super().__init__(saved_locals)

	def do_init(self, kwargs):
		input_shape = self.get_inp_shape()
		self.r, self.c, self.channels = input_shape
		self.fsz = self.r * self.c * self.channels
		self.shape = (None, self.fsz)
		self.param = 0
		self.activation = echo

	def forward(self, inp, training=True):
		return inp.reshape(-1, self.fsz)

	def backprop(self, grads, do_d_inp=True):
		return grads.reshape(-1, self.r, self.c, self.channels)


class Reshape(Layer):
	def __init__(
			self,
			target_shape,
			name=None
			):
		saved_locals = locals()  # save for do_init() function
		super().__init__(saved_locals)

	def do_init(self, kwargs):
		self.input_shape = self.get_inp_shape()
		self.target_shape = kwargs.get('target_shape')
		tt = 1
		for i in self.input_shape:
			tt *= i
		for i in self.target_shape:
			tt /= i
		if tt != 1:
			raise Exception("Cannot Reshape input " + str(self.input_shape) + " to " + str(self.target_shape) + '.')
		self.shape = (None, *self.target_shape)
		self.param = 0
		self.activation = echo

	def forward(self, inp, training=True):
		return inp.reshape(-1, *self.target_shape)

	def backprop(self, grads, do_d_inp=True):
		return grads.reshape(-1, *self.input_shape)
