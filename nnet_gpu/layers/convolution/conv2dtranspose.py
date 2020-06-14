#!/usr/bin/env python3
from .conv2d import *


# TODO: Fix backprop and all, not working right now.

class conv2dtranspose(
		conv2d):  # kernels are flipped of cpu version rn, cpukern = gpukern[:,::-1,::-1,:].transpose(3,1,2,0)
	def __init__(self, num_kernels=0, input_shape=None, kernel_size=0, kernels=None, activation=echo, biases=0,
			stride=(1, 1), dilation=(1, 1), padding=None, batches=1, backp=True, std=0.01, name=None, out_row=None,
			out_col=None):
		super().__init__(num_kernels=num_kernels, input_shape=input_shape, kernel_size=kernel_size, kernels=kernels, activation=activation,
				biases=biases, stride=stride, dilation=dilation, padding=padding, batches=batches, backp=backp, std=std, name=name,
				out_row=out_row, out_col=out_col)

	def cal_padding(self, sz, ksz, stride, dilation):
		oht = self.cal_outsize(sz, ksz, stride, 0, dilation)
		return (stride * (sz - 1) + ksz - oht) // 2

	@property
	def num_kernels(self):
		return self.kernels.shape[0]

	def init_kernel_bias(self, num_inp_channels, kernel_size, num_kernels, mean=0, std=0.01, dtype=cp.float32):
		weights = std * cp.random.randn(num_kernels, kernel_size[0], kernel_size[1], num_inp_channels, dtype=cp.float32) + mean
		# weights/=cp.sqrt(num_inp_channels)
		bias = std * cp.random.randn(1, num_kernels, dtype=cp.float32) + mean
		return weights.astype(dtype, copy=False), bias.astype(dtype, copy=False)

	def init_back(self):
		inp = emptyHelper((self.batches, self.row, self.col, self.channels))
		self.d_ker = conv2d(input_shape=(self.row, self.col, self.batches), kernels=inp, activation=echo, stride=(1, 1),
				dilation=self.stride, padding=self.padding, backp=False, out_row=self.kernel_size[0], out_col=self.kernel_size[1])
		self.d_inp = conv2d(input_shape=(self.out_row, self.out_col, self.num_kernels), kernels=self.kernels, activation=echo,
				stride=self.stride, padding=self.padding, dilation=self.dilation, backp=False, out_row=self.row, out_col=self.col)

	def cal_outsize(self, sz, ksz, stride, pad, dilation=1):
		# dksz = (ksz-1)*dilation + 1		# dilated kernel
		return sz * stride

	def forward(self, inp, training=True):
		"""
		Simply, reverse steps of conv2d.
		Dot product, then col2im.
		"""
		self.inp = inp.transpose(0, 3, 1, 2)
		# inp[batches,channels,row,col]
		self.batches, self.channels, self.row, self.col = self.inp.shape
		coled = cp.tensordot(self.kernels, self.inp, (3, 1))
		coled = cp.moveaxis(coled, 3, 0)  # CAN WE REMOVE THIS SOMEHOW ??
		coled = cp.ascontiguousarray(coled)
		self.z_out = cp.empty((self.batches, self.num_kernels, self.out_row, self.out_col), dtype=coled.dtype)
		col2im(coled.reduced_view(), self.out_row, self.out_col, self.row, self.col,
				self.kernel_size[0], self.kernel_size[1], self.stride[0], self.stride[1], self.padding[0],
				self.padding[1],
				self.dilation[0], self.dilation[1],
				self.z_out)
		self.z_out = self.z_out.transpose(0, 2, 3, 1)
		if self.bias_is_not_0:
			self.z_out = cp.add(self.z_out, self.biases)  # z_out[batches,out_row,out_col,num_kernels]
		self.a_out = self.activation(self.z_out)
		return self.a_out  # a_out[batches,out_row,out_col,num_kernels]

	def backprop(self, grads, layer=1):
		"""
		1.) For kernel gradient (self.d_ker):
				Convolve the saved input as kernel over gradients with stride 1 and dilate the saved input with
				current stride value and current padding.
				The channels are treated as batches and batches as channel so it gives the correct kernel gradient shape.

		2.) For input gradient (self.d_inp):
				Convolution over gradients with self.kernels as kernel. Should give original input shape back.
				All parameters stride,padding,dilation are same as current.

		3.) For biases gradient :
				It's just same as gradient. Just reshape and sum/mean it.
		"""
		if self.activation != echo:
			grads *= self.activation(self.z_out, self.a_out, derivative=True)
		self.d_ker.kernels = self.inp.transpose(0, 2, 3, 1)  # t makes[batches,row,col,channels]
		self.grad_event = stream_maps.default_stream.record(self.grad_event)
		with self.backp_stream:
			self.backp_stream.wait_event(self.grad_event)
			self.d_c_w = self.d_ker.forward(grads.transpose(3, 1, 2, 0))  # [channels,row,col,batches]
		# self.d_c_w/=self.batches		#take mean change over batches
		if layer:
			d_inputs = cp.ascontiguousarray(self.d_inp.forward(grads))
		# assert d_inputs.shape == (self.batches,self.row,self.col,self.channels),f"{(self.batches,self.row,self.col,self.channels)},{d_inputs.shape}"
		else:
			d_inputs = 0
		if self.bias_is_not_0:
			with self.backp_stream:
				self.d_c_b = grads.reshape(-1, self.num_kernels).sum(axis=0, keepdims=True)
		# self.d_c_b=grads.reshape(-1,self.num_kernels).mean(axis=0,keepdims=True)
		return d_inputs
