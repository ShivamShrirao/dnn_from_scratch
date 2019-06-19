#!/usr/bin/env python3
import numpy as np
from functions import *

sd=np.random.randint(1000)
# print(sd)
np.random.seed(sd)	#470
seq_instance=None

class conv2d:
	def __init__(self,num_kernels=0,input_shape=None,kernel_size=0,kernels=None,activation=echo,biases=0,stride=[1,1],padding=0,backp=True,name=None):		#padding=(ksz-1)/2 for same shape in stride 1
		#input_shape[row,col,channels],kernels(channels,ksz,ksz,num_kernels),biases[1,num_ker],stride[row,col]
		if input_shape is None:
			input_shape=seq_instance.get_inp_shape()
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		self.activation=activation
		self.stride=stride
		self.type=self.__class__.__name__
		self.input_shape=input_shape
		self.row,self.col,self.channels=input_shape
		self.batches=1
		self.kernels=kernels
		self.biases=biases
		self.w_m=0
		self.w_v=0
		self.b_m=0
		self.b_v=0
		if self.kernels is None:
			self.kernel_size=kernel_size
			self.num_kernels=num_kernels
			self.kernels,self.biases = self.init_kernel_bias(self.channels,self.kernel_size,self.num_kernels)
		else:
			self.kernel_size=kernels.shape[1]
			self.num_kernels=kernels.shape[3]
		self.kern = self.kernels.reshape(-1,self.num_kernels)
		self.weights = self.kernels
		self.padding=padding
		if not self.padding:							#take care of padding in backprop too
			self.padding=(self.kernel_size-1)//2					#currently don't give 'even' self.kernel_size
		self.out_row,self.out_col=((self.row-self.kernel_size+2*self.padding)//stride[0]+1),((self.col-self.kernel_size+2*self.padding)//stride[1]+1)
		self.prow=self.row+2*self.padding
		self.pcol=self.col+2*self.padding
		self.padded=np.zeros((self.batches,self.channels,self.prow,self.pcol))
		self.param=(self.kernel_size*self.kernel_size*self.channels+1)*self.num_kernels
		# Take all windows into a matrix
		window=(np.arange(self.kernel_size)[:,None]*self.prow+np.arange(self.kernel_size)).ravel()+np.arange(self.channels)[:,None]*self.prow*self.pcol
		slider=(np.arange(self.out_row*stride[0])[:,None]*self.prow+np.arange(self.out_col*stride[1]))
		self.ind = window.ravel()+slider[::stride[0],::stride[1]].ravel()[:,None]
		self.output=np.empty((self.batches,self.out_row*self.out_col,self.num_kernels))
		# bind= np.arange(self.batches)[:,None]*self.channels*self.prow*self.pcol+self.ind.ravel()		#for self.batches
		self.shape=(None,self.out_row,self.out_col,self.num_kernels)
		if backp:
			self.init_back()
	def init_back(self):
		self.flipped=self.kernels[:,::-1,::-1,:].transpose(3,1,2,0)	#self.flipped[num_kernels,self.kernel_size,self.kernel_size,channels]
		pad=(self.kernel_size-1)//2
		errors=self.output.reshape(self.batches,self.out_row,self.out_col,self.num_kernels)
		self.d_ker=conv2d(input_shape=(self.row,self.col,self.batches),kernels=errors,activation=echo,padding=pad,backp=False)
		self.d_inp=conv2d(input_shape=(self.out_row,self.out_col,self.num_kernels),kernels=self.flipped,activation=echo,backp=False)
	def init_kernel_bias(self,num_inp_channels, kernel_size, num_kernels,mean=0,std=0.1):
		shape = [num_inp_channels, kernel_size, kernel_size, num_kernels]
		weights = std*np.random.randn(*shape) + mean
		# weights/=np.sqrt(kernel_size*kernel_size*num_inp_channels)
		bias = std*np.random.randn(1,num_kernels) + mean
		return weights, bias

	def forward(self,inp):
		self.inp=inp.transpose(0,3,1,2)  #inp[batches,channels,row,col]
		batches,channels=self.inp.shape[:2]
		if self.channels!=channels:
			self.channels=channels
			self.padded=np.zeros((self.batches,self.channels,self.prow,self.pcol))
			window=(np.arange(self.kernel_size)[:,None]*self.prow+np.arange(self.kernel_size)).ravel()+np.arange(self.channels)[:,None]*self.prow*self.pcol
			slider=(np.arange(self.out_row*self.stride[0])[:,None]*self.prow+np.arange(self.out_col*self.stride[1]))
			self.ind = window.ravel()+slider[::self.stride[0],::self.stride[1]].ravel()[:,None]
		if self.batches!=batches:
			self.batches=batches
			self.padded=np.zeros((self.batches,self.channels,self.prow,self.pcol))
			self.output=np.empty((self.batches,self.out_row*self.out_col,self.num_kernels))
		self.padded[:,:,self.padding:-self.padding,self.padding:-self.padding]=self.inp
		self.kern=self.kernels.reshape(-1,self.num_kernels)
		for i,img in enumerate(self.padded):		#img[self.channels,self.row,self.col]
			# windows(out_row*out_col, kernel_size*kernel_size*channels) . kernels(channels*kernel_size*kernel_size,num_kernels)
			self.output[i]=np.dot(img.take(self.ind), self.kern)+self.biases
		# output=np.array([(np.dot(np.take(i,self.ind),self.kern)+self.biases) for i in padded]).reshape(self.batches,self.out_row,self.out_col,self.num_kernels)
		# output=(np.dot(np.take(padded, bind).reshape(-1,self.channels*kernel_size*kernel_size), self.kern)+self.biases)
					# [self.batches*self.out_row*self.out_col,self.channels*kernel_size*kernel_size] . [self.channels*kernel_size*kernel_size, self.num_kernels]
		self.z_out=self.output.reshape(self.batches,self.out_row,self.out_col,self.num_kernels)
		self.a_out=self.activation(self.z_out)
		return self.a_out

	def backprop(self,errors,layer=1):								#strides[batch,row,col,depth]
		#errors[batches,esz,esz,num_kernels],inp[batches,row,col,channels],kernels(channels,kernel_size,kernel_size,num_kernels),biases[1,num_kernels],stride[row,col]
		errors*=self.activation(self.z_out,self.a_out,derivative=True)
		self.d_ker.kernels=errors
		self.d_c_w=self.d_ker.forward(self.inp.transpose(1,2,3,0))
		self.d_c_w/=self.batches		#take mean change over batches
		# Backprop for inp.		errors[batches,esz,esz,num_kernels]	self.flipped[num_kernels,kernel_size,kernel_size,channels]
		if layer:
			d_inputs=self.d_inp.forward(errors)
		else:
			d_inputs=0
		self.d_c_b=self.d_ker.kern.mean(axis=0,keepdims=True)

		return d_inputs

class max_pool:
	def __init__(self,input_shape=None,ksize=[2,2],stride=[2,2],name=None):
		#inp[batches,row,col,channels], kernels[ksz,ksz], stride[row,col]
		self.ksz=ksize[0]
		self.param=0
		self.type=self.__class__.__name__
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		if input_shape is None:
			input_shape=seq_instance.get_inp_shape()
		self.batches=1
		self.row,self.col,self.channels=input_shape
		self.rem_col=self.row%self.ksz
		if self.rem_col:
			self.padded=np.zeros((self.batches,self.row,self.col,self.channels))
		self.out_row,self.out_col=self.row//self.ksz,self.col//self.ksz
		self.row-=self.rem_col
		self.col-=self.rem_col
		self.shape=(None,self.out_row,self.out_col,self.channels)
		self.activation=echo

	def forward(self,inp):
		self.input_shape=inp.shape
		batches=self.input_shape[0]
		if self.rem_col:
			inp=inp[:,:-self.rem_col,:-self.rem_col,:]
			if self.batches!=batches:
				self.padded=np.zeros(self.input_shape)
		self.batches=batches
		inp=inp.reshape(self.batches,self.out_row,self.ksz,self.out_col,self.ksz,self.channels)
		output=inp.max(axis=(2,4),keepdims=True)
		self.mask=(inp==output)
		return output.reshape(self.batches,self.out_row,self.out_col,self.channels)

	def backprop(self,errors,layer=1):
		#errors[self.batches,esz,esz,self.channels],inp[self.batches,row,col,self.channels],kernels[self.ksz,self.ksz],stride[row,col]
		z_out=(self.mask*errors.reshape(self.batches,self.out_row,1,self.out_col,1,self.channels))
		if self.rem_col:
			self.padded[:,:-self.rem_col,:-self.rem_col,:]=z_out.reshape(self.batches,self.row,self.col,self.channels)
			return self.padded.reshape(self.input_shape)
		else:
			return z_out.reshape(self.input_shape)

class flatten:
	def __init__(self,name=None):
		self.type=self.__class__.__name__
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		input_shape=seq_instance.get_inp_shape()
		self.r,self.c,self.channels=input_shape
		self.fsz=self.r*self.c*self.channels
		self.shape=(None,self.fsz)
		self.param=0
		self.activation=echo

	def forward(self,inp):
		return inp.reshape(-1,self.fsz)

	def backprop(self,errors,layer=1):
		return errors.reshape(-1,self.r,self.c,self.channels)

class dense:
	def __init__(self,num_out,input_shape=None,activation=echo,mean=0,std=0.1,name=None):
		self.type=self.__class__.__name__
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		if input_shape is None:
			input_shape=seq_instance.get_inp_shape()[0]
		self.activation=activation
		self.weights = std*np.random.randn(input_shape,num_out) + mean
		# weights/=np.sqrt(input_shape)
		self.biases = std*np.random.randn(1,num_out) + mean
		self.kernels = self.weights
		self.w_m=0
		self.w_v=0
		self.b_m=0
		self.b_v=0
		self.shape=(None,num_out)
		self.param=input_shape*num_out + num_out
		self.cross=False

	def forward(self,inp):
		self.inp=inp
		self.z_out=np.dot(inp,self.weights)+self.biases
		self.a_out=self.activation(self.z_out)
		return self.a_out

	def backprop(self,errors,layer=1):
		errors*=self.activation(self.z_out,self.a_out,derivative=True,cross=self.cross)
		d_c_b=errors
		self.d_c_w=np.dot(self.inp.T,d_c_b)/self.inp.shape[0]
		if layer:
			d_c_a=np.dot(d_c_b,self.weights.T)
		else:
			d_c_a=0
		self.d_c_b=d_c_b.mean(axis=0,keepdims=True)
		return d_c_a

class InputLayer:
	def __init__(self,shape):
		self.name='input_layer'
		self.type=self.__class__.__name__
		self.shape=(None,*shape)
		self.param=0
		self.activation=echo

class dropout:
	def __init__(self,name=None):
		self.type=self.__class__.__name__
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		input_shape=seq_instance.get_inp_shape()
		self.r,self.c,self.channels=input_shape
		self.fsz=self.r*self.c*self.channels
		self.shape=(None,self.fsz)
		self.param=0
		self.activation=echo

	def forward(self,inp):
		return inp.reshape(-1,self.fsz)

	def backprop(self,errors,layer=1):
		return errors.reshape(-1,self.r,self.c,self.channels)