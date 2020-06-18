#!/usr/bin/env python3
from ..functions import *
import numpy as np
import cupy as cp


class Layer:
	"""
	The base do_d_inp. All layers are derived from it.
	"""
	saved_parameters = None
	init_done = False
	shape = (None, None)
	input_layer = None
	input_shape = None
	output_layers = []
	bias_is_not_0 = True

	def __init__(self, saved_locals):
		self.name = self.__class__.__name__
		self.type = self.__class__.__name__
		self.dtype = cp.float32
		self.param = 0
		self.activation = echo
		self.saved_parameters = saved_locals
		if self.saved_parameters is not None:
			self.saved_parameters.pop('self')
			name = self.saved_parameters.get('name')
			if name is not None:
				self.name = name
			if self.saved_parameters.get('input_shape') is not None:
				self.do_init(self.saved_parameters)
				self.init_done = True

	def do_init(self, kwargs):
		"""
		Do all the initialization calculations here.
		Set self.init_done = True manually after calling.
		:param kwargs: self.saved_parameters, passed as parameter cause kwargs smaller in length (readability).
		"""
		self.init_done = True

	def __str__(self):
		return self.name + super().__str__()

	def __call__(self, lyr):
		self.input_layer = lyr
		lyr.output_layers.append(self)
		if not self.init_done:
			self.do_init(self.saved_parameters)
			self.init_done = True
		return self

	def get_inp_shape(self):
		return self.input_layer.shape[1:]

	def forward(self, inp, training=True):
		pass

	def backprop(self, grads, do_d_inp=True):
		pass


class InputLayer(Layer):
	"""
	Just placeholder for input do_d_inp.
	"""
	def __init__(self, shape=None):
		super().__init__(None)
		try:
			self.shape = (None, *shape)
		except TypeError:
			self.shape = (None, shape)
