#!/usr/bin/env python3
from .base_layer import *
from ..stream_handler import stream_maps


class Dense(Layer):
	def __init__(
			self,
			num_out,
			input_shape=None,
			weights=None,
			biases=None,
			activation=echo,
			mean=0,
			std=0.01,
			name=None,
			dtype=cp.float32,
			**kwargs
			):
		saved_locals = locals()  # save for do_init() function
		super().__init__(saved_locals)

	def do_init(self, kwargs):
		self.dtype = kwargs.get('dtype')
		self.input_shape = kwargs.get('input_shape')
		if self.input_shape is None:
			self.input_shape = self.get_inp_shape()[0]		# Just the flat number of inputs
		self.activation = kwargs.get('activation')
		weights = kwargs.get('weights')
		biases = kwargs.get('biases')
		std = kwargs.get('std')
		num_out = kwargs.get('num_out')
		mean = kwargs.get('mean')
		if weights is None:
			self.weights = std * cp.random.randn(self.input_shape, num_out, dtype=self.dtype) + mean
		# weights/=np.sqrt(self.input_shape)
		else:
			if weights.shape != (self.input_shape, num_out):
				raise Exception("weights should be of shape: " + str((self.input_shape, num_out)))
			else:
				self.weights = cp.asarray(weights)
		if biases is None:
			self.biases = std * cp.random.randn(1, num_out, dtype=self.dtype) + mean
		else:
			if biases.shape != (1, num_out):
				raise Exception("biases should be of shape: " + str((1, num_out)))
			else:
				self.biases = cp.asarray(biases)
		self.kernels = self.weights
		self.w_m = cp.zeros_like(self.weights, dtype=self.dtype)
		self.w_v = cp.zeros_like(self.weights, dtype=self.dtype)
		self.b_m = cp.zeros_like(self.biases, dtype=self.dtype)
		self.b_v = cp.zeros_like(self.biases, dtype=self.dtype)
		self.shape = (None, num_out)
		self.param = self.input_shape * num_out + num_out
		self.not_softmax_cross_entrp = True
		if self.activation == echo:
			self.notEcho = False
		else:
			self.notEcho = True
		self.backp_stream = stream_maps.get_next_stream()
		self.grad_event = stream_maps.default_stream.record()

	def forward(self, inp, training=True):
		self.inp = inp
		self.z_out = self.inp.dot(self.weights) + self.biases
		self.a_out = self.activation(self.z_out)
		return self.a_out

	def backprop(self, grads, do_d_inp=True):
		if self.notEcho and self.not_softmax_cross_entrp:  # TODO - make it better in future by adding Activation layer in graph
			grads *= self.activation(self.z_out, self.a_out, derivative=True)
		self.grad_event = stream_maps.default_stream.record(self.grad_event)
		with self.backp_stream:
			self.backp_stream.wait_event(self.grad_event)
			self.d_c_w = self.inp.T.dot(grads)#/self.inp.shape[0]
		if do_d_inp:
			d_c_a = grads.dot(self.weights.T)
		else:
			d_c_a = 0
		with self.backp_stream:
			self.d_c_b = grads.sum(axis=0, keepdims=True)
		# self.d_c_b=grads.mean(axis=0,keepdims=True)
		return d_c_a
