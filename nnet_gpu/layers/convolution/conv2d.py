#!/usr/bin/env python3
from ..base_layer import *
from .. import seqinst
from ...stream_handler import stream_maps
from .conv_utils import *

class conv2d(Layer):
	def __init__(self,num_kernels=0, input_shape=None, kernel_size=0, kernels=None, activation=echo, biases=0, stride=(1,1), dilation=(1,1), padding=None, batches=1, backp=True, std=0.01, name=None, out_row=None, out_col=None, off_transpose=0):
		#input_shape[row,col,channels], kernels(channels,ksz[0],ksz[1],num_kernels), biases[1,num_ker], stride[row,col]
		super().__init__()
		if input_shape is None:
			input_shape=seqinst.seq_instance.get_inp_shape()
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		self.activation=activation
		self.dtype=cp.float32
		self.stride=stride
		self.type=self.__class__.__name__
		self.input_shape=input_shape
		self.row,self.col,self.channels=input_shape
		self.batches=batches
		self.kernels=kernels
		self.biases=biases
		if self.kernels is None:
			if np.isscalar(kernel_size):
				self.kernel_size=(kernel_size,kernel_size)
			else:
				self.kernel_size=kernel_size
			self.kernels,self.biases = self.init_kernel_bias(self.channels,self.kernel_size,num_kernels,std=std,dtype=self.dtype)
		else:
			self.kernel_size=kernels.shape[1:3]
		self.weights = self.kernels
		self.bias_is_not_0=True
		if cp.isscalar(self.biases):				# DO BETTER FIX
			if self.biases==0:
				self.bias_is_not_0=False
		self.dilation = dilation
		self.padding = padding
		if padding == None:
			self.padding = self.cal_padding(self.row, self.kernel_size[0], self.stride[0], self.dilation[0]), self.cal_padding(self.col, self.kernel_size[1], self.stride[1], self.dilation[1])
		if out_row is None:
			self.out_row=self.cal_outsize(self.row,self.kernel_size[0],self.stride[0],self.padding[0],self.dilation[0])
		else:
			self.out_row=out_row
		if out_col is None:
			self.out_col=self.cal_outsize(self.row,self.kernel_size[1],self.stride[1],self.padding[1],self.dilation[1])
		else:
			self.out_col=out_col
		self.param=(self.kernel_size[0]*self.kernel_size[1]*self.channels+1)*self.num_kernels
		if backp:
			self.backp_stream=stream_maps.get_next_stream()
			self.grad_event=stream_maps.default_stream.record()
			self.w_m=cp.zeros_like(self.weights,dtype=self.dtype)
			self.w_v=cp.zeros_like(self.weights,dtype=self.dtype)
			if self.bias_is_not_0:
				self.b_m=cp.zeros_like(self.biases,dtype=self.dtype)
				self.b_v=cp.zeros_like(self.biases,dtype=self.dtype)
			self.init_back()

	def cal_padding(self,sz,ksz,stride,dilation):
		return (ksz - 1) // 2

	@property
	def num_kernels(self):
		return self.kernels.shape[3]

	@property
	def shape(self):
		return (None,self.out_row,self.out_col,self.num_kernels)

	# @property
	# def bias_is_not_0(self):
	# 	if cp.isscalar(self.biases):
	# 		if self.biases==0:
	# 			return False
	# 	return True

	def init_back(self):
		global conv2dtranspose
		from .conv2dtranspose import conv2dtranspose
		grads = emptyHelper((self.batches,self.out_row,self.out_col,self.num_kernels))
		self.d_ker=conv2d(input_shape=(self.row,self.col,self.batches),kernels=grads,activation=echo,stride=(1,1),dilation=self.stride,padding=self.padding,backp=False,out_row=self.kernel_size[0],out_col=self.kernel_size[1])
		self.d_inp=conv2dtranspose(input_shape=(self.out_row,self.out_col,self.num_kernels),kernels=self.kernels,activation=echo,stride=self.stride,padding=self.padding,dilation=self.dilation,backp=False,out_row=self.row,out_col=self.col)

	def init_kernel_bias(self,num_inp_channels, kernel_size, num_kernels,mean=0,std=0.01,dtype=cp.float32):
		weights = std*cp.random.randn(num_inp_channels, kernel_size[0], kernel_size[1], num_kernels,dtype=cp.float32) + mean
		# weights/=cp.sqrt(num_inp_channels)
		bias = std*cp.random.randn(1,num_kernels,dtype=cp.float32) + mean
		return weights.astype(dtype,copy=False), bias.astype(dtype,copy=False)

	def cal_outsize(self,sz,ksz,stride,pad,dilation=1):
		dksz = (ksz-1)*dilation + 1		# dilated kernel
		return (sz + 2*pad - dksz)//stride + 1

	def forward(self,inp,training=True):
		"""
		Simple, just do im2col and then dot product.
		"""
		inp=cp.ascontiguousarray(inp.transpose(0,3,1,2))
		self.inp=inp
		#inp[batches,channels,row,col]
		self.batches,self.channels,self.row,self.col=self.inp.shape
		coled = cp.empty((self.batches, self.channels, self.kernel_size[0], self.kernel_size[1], self.out_row, self.out_col), dtype=self.dtype)
		im2col(self.inp.reduced_view(), self.row, self.col, self.out_row, self.out_col,
				self.kernel_size[0], self.kernel_size[1], self.stride[0], self.stride[1], self.padding[0], self.padding[1],
				self.dilation[0], self.dilation[1],
				coled)
		self.z_out = cp.tensordot(coled, self.kernels, ((1, 2, 3), (0, 1, 2)))
		if self.bias_is_not_0:
			self.z_out=cp.add(self.z_out,self.biases)
		assert self.z_out.shape==(self.batches,self.out_row,self.out_col,self.num_kernels)
		self.a_out=self.activation(self.z_out)
		return self.a_out				# a_out[self.batches,self.out_row,self.out_col,self.num_kernels]

	def backprop(self,grads,layer=1):								#strides[batch,row,col,depth]
		"""
		grads[batches,esz,esz,num_kernels],inp[batches,channels,row,col],kernels(channels,ksz,ksz,num_kernels),biases[1,num_kernels]
		1.) For kernel gradient (self.d_ker):
				Convolve the gradients as kernel over saved input with stride 1 and dilate the gradient with
				current stride value and current padding.
				The channels are treated as batches and batches as channel so it gives the correct kernel gradient shape.

		2.) For input gradient (self.d_inp):
				Transposed convolution over gradients with self.kernels as kernel. Should give original input shape back.
				All parameters stride,padding,dilation are same as current.

		3.) For biases gradient :
				It's just same as gradient. Just reshape and sum/mean it.

		TODO: Compare difference of sum and mean for bias.
		"""
		if self.activation != echo:
			grads*=self.activation(self.z_out,self.a_out,derivative=True)
		self.d_ker.kernels=grads 						# set gradients as kernel
		self.grad_event=stream_maps.default_stream.record(self.grad_event)
		with self.backp_stream:
			self.backp_stream.wait_event(self.grad_event)
			self.d_c_w=self.d_ker.forward(self.inp.transpose(1,2,3,0))	#[channels,row,col,batches]
		# self.d_c_w/=self.batches		#take mean change over batches
		# Backprop for inp.	grads[batches,esz,esz,num_kernels]	self.flipped[num_kernels,kernel_size[0],kernel_size[1],channels]
		if layer:
			d_inputs=cp.ascontiguousarray(self.d_inp.forward(grads))
			assert d_inputs.shape == (self.batches,self.row,self.col,self.channels),f"{(self.batches,self.row,self.col,self.channels)},{d_inputs.shape}"
		else:
			d_inputs=0
		if self.bias_is_not_0:
			with self.backp_stream:
				self.d_c_b=grads.reshape(-1,self.num_kernels).sum(axis=0,keepdims=True)
				# self.d_c_b=grads.reshape(-1,self.num_kernels).mean(axis=0,keepdims=True)
		return d_inputs