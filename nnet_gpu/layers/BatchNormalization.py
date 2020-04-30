#!/usr/bin/env python3
from .base_layer import *
from . import seqinst
from ..stream_handler import stream_maps

class BatchNormalization(Layer):
	def __init__(self,momentum=0.9,epsilon=1e-10,name=None):
		super().__init__()
		self.dtype=cp.float32
		self.type=self.__class__.__name__
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		input_shape=seqinst.seq_instance.get_inp_shape()
		self.shape=(None,*input_shape)
		self.batches=1
		self.inp_shape=(self.batches,*input_shape)
		self.biases=cp.zeros(input_shape,dtype=self.dtype)		#biases is beta
		self.weights=cp.ones(input_shape,dtype=self.dtype)		#weights is gamma
		self.gamma=self.weights
		self.beta=self.biases
		self.kernels = self.weights
		self.w_m=cp.zeros_like(self.weights,dtype=self.dtype)
		self.w_v=cp.zeros_like(self.weights,dtype=self.dtype)
		self.b_m=cp.zeros_like(self.biases,dtype=self.dtype)
		self.b_v=cp.zeros_like(self.biases,dtype=self.dtype)
		self.epsilon=epsilon
		self.momentum=momentum
		self.moving_mean=None
		self.moving_var=None
		self.param=4*input_shape[-1]
		self.activation=echo
		self.backp_stream=stream_maps.get_next_stream()
		self.grad_event=stream_maps.default_stream.record()

	def forward(self,inp,training=True):		# yeah, I know, too many repetitions
		#inp[batches,row,col,channels]
		if training:
			self.inp_shape=inp.shape
			mean=inp.mean(axis=0)					#(row,col,channels)
			self.xmu=inp-mean 						#(batches,row,col,channels)
			var=(self.xmu**2).mean(axis=0)			#(row,col,channels)
			self.ivar=1/(var+self.epsilon)			#(row,col,channels)
			self.istd=cp.sqrt(self.ivar)				#(row,col,channels)
			self.xnorm=self.xmu*self.istd 				#(batches,row,col,channels)
			if self.moving_mean is None:
				self.moving_mean=mean
				self.moving_var=var
			else:
				with self.backp_stream:
					self.moving_mean=self.momentum*self.moving_mean + (1-self.momentum)*mean
					self.moving_var=self.momentum*self.moving_var + (1-self.momentum)*var
		else:
			if self.moving_mean is None:
				self.inp_shape=inp.shape
				mean=inp.mean(axis=0)					#(row,col,channels)
				self.xmu=inp-mean 						#(batches,row,col,channels)
				var=(self.xmu**2).mean(axis=0)			#(row,col,channels)
				self.ivar=1/(var+self.epsilon)			#(row,col,channels)
				self.istd=cp.sqrt(self.ivar)				#(row,col,channels)
				self.moving_mean=mean
				self.moving_var=var
				self.xnorm=self.xmu*self.istd 				#(batches,row,col,channels)
			else:
				self.inp_shape=inp.shape
				# self.xmu=inp								#(batches,row,col,channels)	## all this is just for proper shape while model.free()
				# self.istd=self.moving_var					#(row,col,channels)
				self.xnorm=(inp-self.moving_mean)/cp.sqrt(self.moving_var+self.epsilon)
		return self.xnorm*self.weights+self.biases

	def backprop(self,grads,layer=1):
		#grads(batches,row,col,channels), xmu(batches,row,col,channels)=inp-mean 		#FU
		batches=self.inp_shape[0]
		if batches!=self.batches:
			self.batches=batches

		self.d_c_b=grads.sum(axis=0) 				#(row,col,channels)		# biases is beta
		self.grad_event=stream_maps.default_stream.record(self.grad_event)

		with self.backp_stream:
			self.backp_stream.wait_event(self.grad_event)
			self.d_c_w=(self.xnorm*grads).sum(axis=0)	#(row,col,channels)		# gamma is weights

		d_inp=(1/self.batches)*self.istd*self.weights*(self.batches*grads-self.d_c_b-self.xmu*self.ivar*((grads*self.xmu).sum(axis=0)))
		# d_inp=self.istd*self.weights*(self.batches*grads-self.d_c_b-self.xmu*self.ivar*((grads*self.xmu).sum(axis=0)))
		return d_inp