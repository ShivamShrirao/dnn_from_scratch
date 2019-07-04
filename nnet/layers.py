#!/usr/bin/env python3
import numpy as np
from nnet.functions import *
from ctypes import CDLL,c_int,c_void_p

ctake=CDLL("libctake.so")	# gcc ctake_threaded.c -fPIC -shared -o libctake.so -O3 -lpthread
NUM_THREADS = 4

sd=np.random.randint(1000)
print("Seed:",sd)
np.random.seed(sd)
seq_instance=None

class conv2d:
	def __init__(self,num_kernels=0,input_shape=None,kernel_size=0,kernels=None,activation=echo,biases=0,stride=[1,1],padding=0,backp=True,std=0.01,name=None):		#padding=(ksz-1)/2 for same shape in stride 1
		#input_shape[row,col,channels],kernels(channels,ksz,ksz,num_kernels),biases[1,num_ker],stride[row,col]
		if input_shape is None:
			input_shape=seq_instance.get_inp_shape()
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		self.activation=activation
		self.dtype=np.float32
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
			self.kernels,self.biases = self.init_kernel_bias(self.channels,self.kernel_size,self.num_kernels,std=std)
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
		self.padded=np.zeros((self.batches,self.channels,self.prow,self.pcol),dtype=self.dtype)
		self.param=(self.kernel_size*self.kernel_size*self.channels+1)*self.num_kernels
		# Take all windows into a matrix
		window=(np.arange(self.kernel_size)[:,None]*self.prow+np.arange(self.kernel_size)).ravel()+np.arange(self.channels)[:,None]*self.prow*self.pcol
		slider=(np.arange(self.out_row*stride[0])[:,None]*self.prow+np.arange(self.out_col*stride[1]))
		self.ind = window.ravel()+slider[::stride[0],::stride[1]].ravel()[:,None]
		self.output=np.empty((self.batches,self.out_row*self.out_col,self.num_kernels),dtype=self.dtype)
		self.coled=np.empty((self.batches,*self.ind.shape),dtype=np.float32).reshape(-1,self.channels*self.kernel_size*self.kernel_size)
		# bind= np.arange(self.batches)[:,None]*self.channels*self.prow*self.pcol+self.ind.ravel()		#for self.batches
		self.shape=(None,self.out_row,self.out_col,self.num_kernels)
		if backp:
			self.init_back()
	def init_back(self):				# flipped kernel has same reference as original one so it will be updated automatically with original kernel
		self.flipped=self.kernels[:,::-1,::-1,:].transpose(3,1,2,0)	#flipped[num_kernels,kernel_size,kernel_size,channels]
		pad=(self.kernel_size-1)//2
		errors=self.output.reshape(self.batches,self.out_row,self.out_col,self.num_kernels)
		self.d_ker=conv2d(input_shape=(self.row,self.col,self.batches),kernels=errors,activation=echo,padding=pad,backp=False)
		self.d_inp=conv2d(input_shape=(self.out_row,self.out_col,self.num_kernels),kernels=self.flipped,activation=echo,backp=False)
	def init_kernel_bias(self,num_inp_channels, kernel_size, num_kernels,mean=0,std=0.01):
		shape = [num_inp_channels, kernel_size, kernel_size, num_kernels]
		weights = std*np.random.randn(*shape) + mean
		# weights/=np.sqrt(num_inp_channels)
		bias = std*np.random.randn(1,num_kernels) + mean
		return weights.astype(self.dtype), bias.astype(self.dtype)

	def forward(self,inp,training=True):
		self.inp=inp.transpose(0,3,1,2)  #inp[batches,channels,row,col]
		batches,channels=self.inp.shape[:2]
		if self.channels!=channels:
			self.channels=channels
			self.padded=np.zeros((self.batches,self.channels,self.prow,self.pcol),dtype=self.dtype)
			window=(np.arange(self.kernel_size)[:,None]*self.prow+np.arange(self.kernel_size)).ravel()+np.arange(self.channels)[:,None]*self.prow*self.pcol
			slider=(np.arange(self.out_row*self.stride[0])[:,None]*self.prow+np.arange(self.out_col*self.stride[1]))
			self.ind = window.ravel()+slider[::self.stride[0],::self.stride[1]].ravel()[:,None]
			self.coled=np.empty((self.batches,*self.ind.shape),dtype=np.float32).reshape(-1,self.channels*self.kernel_size*self.kernel_size)
		if self.batches!=batches:
			self.batches=batches
			self.padded=np.zeros((self.batches,self.channels,self.prow,self.pcol),dtype=self.dtype)
			# self.output=np.empty((self.batches,self.out_row*self.out_col,self.num_kernels),dtype=self.dtype)
			self.coled=np.empty((self.batches,*self.ind.shape),dtype=np.float32).reshape(-1,self.channels*self.kernel_size*self.kernel_size)
		self.padded[:,:,self.padding:-self.padding,self.padding:-self.padding]=self.inp
		self.kern=self.kernels.reshape(-1,self.num_kernels)
		# for i,img in enumerate(self.padded):		#img[self.channels,self.row,self.col]
			# windows(out_row*out_col, kernel_size*kernel_size*channels) . kernels(channels*kernel_size*kernel_size,num_kernels)
			# self.output[i]=np.dot(img.take(self.ind), self.kern)
		# output=np.array([(np.dot(np.take(i,self.ind),self.kern)+self.biases) for i in padded]).reshape(self.batches,self.out_row,self.out_col,self.num_kernels)
		# output=(np.dot(np.take(padded, bind).reshape(-1,self.channels*kernel_size*kernel_size), self.kern)+self.biases)
					# [self.batches*self.out_row*self.out_col,self.channels*kernel_size*kernel_size] . [self.channels*kernel_size*kernel_size, self.num_kernels]
		ctake.take(c_void_p(self.padded.ctypes.data),c_void_p(self.ind.ctypes.data),c_void_p(self.coled.ctypes.data),c_int(self.batches),c_int(self.padded[0].size),c_int(self.ind.size),c_int(NUM_THREADS))
		self.output=self.coled.dot(self.kern)
		if self.biases is not 0:
			self.output+=self.biases
		self.z_out=self.output.reshape(self.batches,self.out_row,self.out_col,self.num_kernels)
		self.a_out=self.activation(self.z_out)
		return self.a_out

	def backprop(self,errors,layer=1):								#strides[batch,row,col,depth]
		#errors[batches,esz,esz,num_kernels],inp[batches,row,col,channels],kernels(channels,kernel_size,kernel_size,num_kernels),biases[1,num_kernels],stride[row,col]
		if self.activation != echo:
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
		self.dtype=np.float32
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
			self.padded=np.zeros((self.batches,self.row,self.col,self.channels),dtype=self.dtype)
		self.out_row,self.out_col=self.row//self.ksz,self.col//self.ksz
		self.row-=self.rem_col
		self.col-=self.rem_col
		self.shape=(None,self.out_row,self.out_col,self.channels)
		self.activation=echo

	def forward(self,inp,training=True):
		self.input_shape=inp.shape
		batches=self.input_shape[0]
		if self.rem_col:
			inp=inp[:,:-self.rem_col,:-self.rem_col,:]
			if self.batches!=batches:
				self.padded=np.zeros(self.input_shape,dtype=self.dtype)
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
		self.dtype=np.float32
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

	def forward(self,inp,training=True):
		return inp.reshape(-1,self.fsz)

	def backprop(self,errors,layer=1):
		return errors.reshape(-1,self.r,self.c,self.channels)

class dense:
	def __init__(self,num_out,input_shape=None,activation=echo,mean=0,std=0.01,name=None):
		self.dtype=np.float32
		self.type=self.__class__.__name__
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		if input_shape is None:
			input_shape=seq_instance.get_inp_shape()[0]
		self.activation=activation
		self.weights = std*np.random.randn(input_shape,num_out).astype(self.dtype) + mean
		# weights/=np.sqrt(input_shape)
		self.biases = std*np.random.randn(1,num_out).astype(self.dtype) + mean
		self.kernels = self.weights
		self.w_m=0
		self.w_v=0
		self.b_m=0
		self.b_v=0
		self.shape=(None,num_out)
		self.param=input_shape*num_out + num_out
		self.cross_entrp=False

	def forward(self,inp,training=True):
		self.inp=inp
		self.z_out=np.dot(inp,self.weights)+self.biases
		self.a_out=self.activation(self.z_out)
		return self.a_out

	def backprop(self,errors,layer=1):
		if (self.activation!=echo) and (not self.cross_entrp):
			errors*=self.activation(self.z_out,self.a_out,derivative=True)
		d_c_b=errors
		self.d_c_w=np.dot(self.inp.T,d_c_b)/self.inp.shape[0]
		if layer:
			d_c_a=np.dot(d_c_b,self.weights.T)
		else:
			d_c_a=0
		self.d_c_b=d_c_b.mean(axis=0,keepdims=True)
		return d_c_a

class dropout:
	def __init__(self,rate=0.2,name=None):
		self.dtype=np.float32
		self.type=self.__class__.__name__
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		input_shape=seq_instance.get_inp_shape()
		self.shape=(None,*input_shape)
		self.batches=1
		self.rate=rate
		self.scale=1/(1-rate)
		self.mask=np.random.random((self.batches,*input_shape))>self.rate
		self.param=0
		self.activation=echo

	def forward(self,inp,training=True):
		if training:
			self.mask=self.scale*np.random.random(inp.shape)>self.rate 		#generate mask with rate probability
			return inp*self.mask
		else:
			self.mask=inp
			return inp

	def backprop(self,errors,layer=1):
		return errors*self.mask

class BatchNormalization:					#Have to add references to each brah
	def __init__(self,momentum=0.9,epsilon=1e-10,name=None):
		self.dtype=np.float32
		self.type=self.__class__.__name__
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		input_shape=seq_instance.get_inp_shape()
		self.shape=(None,*input_shape)
		self.batches=1
		self.inp_shape=(self.batches,*input_shape)
		self.biases=np.zeros(input_shape).astype(self.dtype)		#biases is beta
		self.weights=np.ones(input_shape).astype(self.dtype)		#weights is gamma
		self.gamma=self.weights
		self.beta=self.biases
		self.kernels = self.weights
		self.w_m=0
		self.w_v=0
		self.b_m=0
		self.b_v=0
		self.epsilon=epsilon
		self.momentum=momentum
		self.moving_mean=None
		self.moving_var=None
		self.param=4*input_shape[-1]
		self.activation=echo

	def forward(self,inp,training=True):		# yeah, I know, too many repetitions
		#inp[batches,row,col,channels]
		if training:
			self.inp_shape=inp.shape
			self.mean=inp.mean(axis=0)					#(row,col,channels)
			self.xmu=inp-self.mean 						#(batches,row,col,channels)
			self.var=(self.xmu**2).mean(axis=0)			#(row,col,channels)
			self.ivar=1/(self.var+self.epsilon)			#(row,col,channels)
			self.istd=np.sqrt(self.ivar)				#(row,col,channels)
			self.xnorm=self.xmu*self.istd 				#(batches,row,col,channels)
			if self.moving_mean is None:
				self.moving_mean=self.mean
				self.moving_var=self.var
			else:
				self.moving_mean=self.momentum*self.moving_mean + (1-self.momentum)*self.mean
				self.moving_var=self.momentum*self.moving_var + (1-self.momentum)*self.var
		else:
			if self.moving_mean is None:
				self.inp_shape=inp.shape
				self.mean=inp.mean(axis=0)					#(row,col,channels)
				self.xmu=inp-self.mean 						#(batches,row,col,channels)
				self.var=(self.xmu**2).mean(axis=0)			#(row,col,channels)
				self.ivar=1/(self.var+self.epsilon)			#(row,col,channels)
				self.istd=np.sqrt(self.ivar)				#(row,col,channels)
				self.moving_mean=self.mean
				self.moving_var=self.var
				self.xnorm=self.xmu*self.istd 				#(batches,row,col,channels)
			else:
				self.inp_shape=inp.shape
				self.xmu=inp								#(batches,row,col,channels)	## all this is just for proper shape while model.free()
				self.istd=self.moving_var					#(row,col,channels)
				self.xnorm=(inp-self.moving_mean)/np.sqrt(self.moving_var+self.epsilon)
		return self.xnorm*self.weights+self.biases

	def backprop(self,errors,layer=1):
		#errors(batches,row,col,channels), xmu(batches,row,col,channels)=inp-mean 		#FU
		batches=self.inp_shape[0]
		if batches!=self.batches:
			self.batches=batches
		self.d_c_b=errors.sum(axis=0) 				#(row,col,channels)		# biases is beta
		self.d_c_w=(self.xnorm*errors).sum(axis=0)	#(row,col,channels)		# gamma is weights
		d_inp=(1/self.batches)*self.istd*self.weights*(self.batches*errors-self.d_c_b-self.xmu*self.ivar*((errors*self.xmu).sum(axis=0)))
		return d_inp

class Activation:
	def __init__(self,activation=echo,input_shape=None,name=None):
		self.dtype=np.float32
		self.type=self.__class__.__name__
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		if input_shape is None:
			input_shape=seq_instance.get_inp_shape()[0]
		self.activation=activation
		self.shape=(None,*input_shape)
		self.param=0
		self.cross_entrp=False

	def forward(self,inp,training=True):
		self.z_out=inp
		self.a_out=self.activation(self.z_out)
		return self.a_out

	def backprop(self,errors,layer=1):
		if (self.activation!=echo) and (not self.cross_entrp):
			errors*=self.activation(self.z_out,self.a_out,derivative=True)
		return errors

class InputLayer:
	def __init__(self,shape):		#just placeholder
		self.name='input_layer'
		self.type=self.__class__.__name__
		self.dtype=np.float32
		self.shape=(None,*shape)
		self.param=0
		self.activation=echo