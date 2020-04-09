#!/usr/bin/env python3
import numpy as np
from nnet.functions import *
from nnet.coled_tracker import coled_tracker
from ctypes import CDLL,c_int,c_void_p
import os
import sys

ctake=CDLL(os.path.join(os.path.dirname(__file__),"libctake.so"))	# gcc nnet/ctake_threaded.c -fPIC -shared -o nnet/libctake.so -O3 -lpthread
if sys.platform == 'win32':
	NUM_THREADS = int(os.environ['NUMBER_OF_PROCESSORS'])
else:
	NUM_THREADS = int(os.popen("nproc").read())

sd=np.random.randint(1000)
print("Seed:",sd)
np.random.seed(sd)
seq_instance=None		# fix this. It's same for multiple models. Will be fixed by self.previous_layer, or a dict for each instance
COLT=coled_tracker()

""" 
Prolly make parallel C code to pad faster.
"""

class Layer:
	def __init__(self):
		self.name=self.__class__.__name__
		self.type=self.__class__.__name__
		self.dtype=np.float32
		self.param=0
		self.activation=echo
		self.input_layer=None
		self.output_layers=[]

	def __str__(self):
		return self.name+super().__str__()

	def __call__(self,lyr):
		self.input_layer=lyr
		lyr.output_layers.append(self)
		return self

class conv2d(Layer):
	def __init__(self,num_kernels=0,input_shape=None,kernel_size=0,kernels=None,activation=echo,biases=0,stride=[1,1],dilation=[1,1],dlate=[1,1],padding=None,batches=1,backp=True,std=0.01,name=None,out_row=None,out_col=None,off_transpose=0):		#padding=(ksz-1)/2 for same shape in stride 1
		#input_shape[row,col,channels],kernels(channels,ksz,ksz,num_kernels),biases[1,num_ker],stride[row,col]
		super().__init__()
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
		self.batches=batches
		self.kernels=kernels
		self.biases=biases
		self.w_m=0
		self.w_v=0
		self.b_m=0
		self.b_v=0
		self.d_c_b=0
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
		self.dilation=dilation
		self.dlate=dlate
		self.erow=self.row+(self.row-1)*(self.dlate[0]-1)
		self.ecol=self.col+(self.col-1)*(self.dlate[1]-1)
		if self.padding is None:							#take care of padding in backprop too
			self.padding=(self.kernel_size-1)//2					#currently don't give 'even' kernel_size
		if out_row is None:
			self.out_row=(self.erow-self.kernel_size+2*self.padding-(self.kernel_size-1)*(self.dilation[0]-1))//stride[0]+1
		else:
			self.out_row=out_row
		if out_col is None:
			self.out_col=(self.ecol-self.kernel_size+2*self.padding-(self.kernel_size-1)*(self.dilation[0]-1))//stride[1]+1
		else:
			self.out_col=out_col
		self.prow=self.erow+2*self.padding
		self.pcol=self.ecol+2*self.padding
		self.padded=np.zeros((self.batches,self.channels,self.prow,self.pcol),dtype=self.dtype)
		self.param=(self.kernel_size*self.kernel_size*self.channels+1)*self.num_kernels
		# Take all windows into a matrix
		self.dksz=self.kernel_size+(self.kernel_size-1)*(self.dilation[0]-1)
		self.off_transpose=off_transpose
		if (self.stride[0]+self.stride[1])>2:
			if backp:
				if self.off_transpose==0:
					cuut=self.padding-self.padding//self.stride[0]
					self.off_transpose=(self.row+2*self.padding)*cuut+cuut
		window=(np.arange(self.dksz,step=self.dilation[0])[:,None]*self.prow+np.arange(self.dksz,step=self.dilation[1])).ravel()+np.arange(self.channels)[:,None]*self.prow*self.pcol+self.off_transpose
		slider=(np.arange(self.out_row*stride[0])[:,None]*self.prow+np.arange(self.out_col*stride[1]))
		self.ind = window.ravel()+slider[::stride[0],::stride[1]].ravel()[:,None]
		self.output=np.empty((self.batches,self.out_row*self.out_col,self.num_kernels),dtype=self.dtype)
		# self.coled=np.empty((self.batches,*self.ind.shape),dtype=self.dtype).reshape(-1,self.channels*self.kernel_size*self.kernel_size)
		self.coled=COLT.alloc(self.ind.size*self.batches,self).reshape(-1,self.channels*self.kernel_size*self.kernel_size)
		COLT.free()
		# bind= np.arange(self.batches)[:,None]*self.channels*self.prow*self.pcol+self.ind.ravel()		#for self.batches
		self.shape=(None,self.out_row,self.out_col,self.num_kernels)
		self.is_not_dker=True
		self.bias_is_not_0=True
		if np.isscalar(self.biases):
			if self.biases==0:
				self.bias_is_not_0=False
		if backp:
			self.init_back()

	def init_back(self):				# flipped kernel has same reference as original one so it will be updated automatically with original kernel
		self.flipped=self.kernels[:,::-1,::-1,:].transpose(3,1,2,0)	#flipped[num_kernels,kernel_size,kernel_size,channels]
		if (self.stride[0]+self.stride[1])>2:
			padk=self.padding
			padi=self.kernel_size-1
			distride=[1,1]
			off_transpose_ker=self.off_transpose
			off_transpose_inp=0
		elif (self.dlate[0]+self.dlate[1])>2:
			padk=self.padding
			padi=self.kernel_size-1
			distride=self.dlate
			off_transpose_ker=0
			off_transpose_inp=(self.out_row+2*padi)*padi+padi
		else:
			padk=padi=(self.kernel_size-1)//2
			distride=self.stride
			off_transpose_ker=off_transpose_inp=0
		grads=self.output.reshape(self.batches,self.out_row,self.out_col,self.num_kernels)
		self.d_ker=conv2d(input_shape=(self.row,self.col,self.batches),kernels=grads,activation=echo,dilation=self.stride,dlate=self.dlate,padding=padk,backp=False,off_transpose=off_transpose_ker,out_row=self.kernel_size,out_col=self.kernel_size,batches=self.channels)
		self.d_ker.is_not_dker=False
		# self.d_ker.dlate=self.dlate
		self.d_inp=conv2d(input_shape=(self.out_row,self.out_col,self.num_kernels),kernels=self.flipped,activation=echo,stride=distride,dlate=self.stride,padding=padi,off_transpose=off_transpose_inp,backp=False,out_row=self.row,out_col=self.col)

	def init_kernel_bias(self,num_inp_channels, kernel_size, num_kernels,mean=0,std=0.01):
		weights = std*np.random.randn(num_inp_channels, kernel_size, kernel_size, num_kernels) + mean
		# weights/=np.sqrt(num_inp_channels)
		bias = std*np.random.randn(1,num_kernels) + mean
		return weights.astype(self.dtype), bias.astype(self.dtype)

	def forward(self,inp,training=True):
		self.inp=inp.transpose(0,3,1,2)
		#inp[batches,channels,row,col]
		batches,channels=self.inp.shape[:2]
		if (self.channels!=channels) or (self.batches!=batches):
			self.channels=channels
			self.batches=batches
			self.padded=np.zeros((self.batches,self.channels,self.prow,self.pcol),dtype=self.dtype)
			self.dksz=self.kernel_size+(self.kernel_size-1)*(self.dilation[0]-1)
			window=(np.arange(self.dksz,step=self.dilation[0])[:,None]*self.prow+np.arange(self.dksz,step=self.dilation[1])).ravel()+np.arange(self.channels)[:,None]*self.prow*self.pcol+self.off_transpose
			slider=(np.arange(self.out_row*self.stride[0])[:,None]*self.prow+np.arange(self.out_col*self.stride[1]))
			self.ind = window.ravel()+slider[::self.stride[0],::self.stride[1]].ravel()[:,None]
			# self.coled=np.empty((self.batches,*self.ind.shape),dtype=self.dtype).reshape(-1,self.channels*self.kernel_size*self.kernel_size)
			self.coled=COLT.alloc(self.ind.size*self.batches,self).reshape(-1,self.channels*self.kernel_size*self.kernel_size)
			COLT.free()
			if not self.is_not_dker:
				if self.padding:
					self.padded[:,:,self.padding:-self.padding:self.dlate[0],self.padding:-self.padding:self.dlate[1]]=self.inp 	# this takes time. FIX 
				else:
					self.padded[:,:,::self.dlate[0],::self.dlate[1]]=self.inp
		if self.is_not_dker:
			if self.padding:
				self.padded[:,:,self.padding:-self.padding:self.dlate[0],self.padding:-self.padding:self.dlate[1]]=self.inp 	# this takes time. FIX 
			else:
				self.padded[:,:,::self.dlate[0],::self.dlate[1]]=self.inp
		self.kern=self.kernels.reshape(-1,self.num_kernels)
		# for i,img in enumerate(self.padded):		#img[self.channels,self.row,self.col]
			# windows(out_row*out_col, kernel_size*kernel_size*channels) . kernels(channels*kernel_size*kernel_size,num_kernels)
			# self.output[i]=np.dot(img.take(self.ind), self.kern)
		# output=np.array([(np.dot(np.take(i,self.ind),self.kern)+self.biases) for i in padded]).reshape(self.batches,self.out_row,self.out_col,self.num_kernels)
		# output=(np.dot(np.take(padded, bind).reshape(-1,self.channels*kernel_size*kernel_size), self.kern)+self.biases)
					# [self.batches*self.out_row*self.out_col,self.channels*kernel_size*kernel_size] . [self.channels*kernel_size*kernel_size, self.num_kernels]
		ctake.take(c_void_p(np.ascontiguousarray(self.padded).ctypes.data),c_void_p(self.ind.ctypes.data),c_void_p(self.coled.ctypes.data),c_int(self.batches),c_int(self.padded[0].size),c_int(self.ind.size),c_int(NUM_THREADS))
		self.output=self.coled.dot(self.kern)
		if self.bias_is_not_0:
			self.output+=self.biases
		self.z_out=self.output.reshape(self.batches,self.out_row,self.out_col,self.num_kernels)
		self.a_out=self.activation(self.z_out)
		return self.a_out

	def backprop(self,grads,layer=1):								#strides[batch,row,col,depth]
		#grads[batches,esz,esz,num_kernels],inp[batches,channels,row,col],kernels(channels,kernel_size,kernel_size,num_kernels),biases[1,num_kernels],stride[row,col]
		if self.activation != echo:
			grads*=self.activation(self.z_out,self.a_out,derivative=True)
		self.d_ker.kernels=grads
		self.d_ker.padded=np.ascontiguousarray(self.padded.transpose(1,0,2,3))
		self.d_c_w=self.d_ker.forward(self.inp.transpose(1,2,3,0))
		# self.d_c_w/=self.batches		#take mean change over batches
		# Backprop for inp.		grads[batches,esz,esz,num_kernels]	self.flipped[num_kernels,kernel_size,kernel_size,channels]
		if layer:
			d_inputs=self.d_inp.forward(grads)
		else:
			d_inputs=0
		if self.bias_is_not_0:
			self.d_c_b=self.d_ker.kern.sum(axis=0,keepdims=True)
		# self.d_c_b=self.d_ker.kern.mean(axis=0,keepdims=True)
		return d_inputs

class conv2dtranspose(conv2d):
	def __init__(self,num_kernels=0,input_shape=None,kernel_size=0,kernels=None,activation=echo,biases=0,stride=[2,2],dilation=[1,1],dlate=[1,1],padding=None,batches=1,backp=True,std=0.01,name=None,out_row=None,out_col=None):
		if input_shape is None:
			input_shape=seq_instance.get_inp_shape()
		out_row=stride[0]*input_shape[0]
		out_col=stride[1]*input_shape[1]
		if (stride[0]+stride[1])>2:
			dlate=stride
			stride=[1,1]
			if padding is None:
				padding=kernel_size-1
		super().__init__(num_kernels=num_kernels,input_shape=input_shape,kernel_size=kernel_size,kernels=kernels,activation=activation,biases=biases,stride=stride,dilation=dilation,dlate=dlate,padding=padding,batches=batches,backp=backp,std=std,name=name,out_row=out_row,out_col=out_col)
		

class max_pool(Layer):
	def __init__(self,input_shape=None,ksize=[2,2],stride=[2,2],name=None):
		#inp[batches,row,col,channels], kernels[ksz,ksz], stride[row,col]
		super().__init__()
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

	def backprop(self,grads,layer=1):
		#grads[self.batches,esz,esz,self.channels],inp[self.batches,row,col,self.channels],kernels[self.ksz,self.ksz],stride[row,col]
		z_out=(self.mask*grads.reshape(self.batches,self.out_row,1,self.out_col,1,self.channels))
		if self.rem_col:
			self.padded[:,:-self.rem_col,:-self.rem_col,:]=z_out.reshape(self.batches,self.row,self.col,self.channels)
			return self.padded.reshape(self.input_shape)
		else:
			return z_out.reshape(self.input_shape)

class globalAveragePool(Layer):
	def __init__(self,input_shape=None,name=None):
		super().__init__()
		self.type=self.__class__.__name__
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		if input_shape is None:
			input_shape=seq_instance.get_inp_shape()
		self.param=0
		self.batches=1
		self.row,self.col,self.channels=input_shape
		self.Ncount=self.row*self.col
		self.shape=(None,self.channels)
		self.activation=echo

	def forward(self,inp,training=True):
		self.input_shape=inp.shape
		self.batches=self.input_shape[0]
		inp=inp.reshape(self.batches,self.Ncount,self.channels)
		output=inp.mean(axis=1)
		return output.reshape(self.batches,self.channels)

	def backprop(self,grads,layer=1):
		# grads/=self.Ncount
		z_out=grads.repeat(self.Ncount,axis=0).reshape(self.batches,self.row,self.col,self.channels)
		return z_out

class upsampling(Layer):
	def __init__(self,input_shape=None,ksize=[2,2],stride=[2,2],name=None):
		#inp[batches,row,col,channels], kernels[ksz,ksz], stride[row,col]
		super().__init__()
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
		self.out_row,self.out_col=self.row*self.ksz,self.col*self.ksz
		self.shape=(None,self.out_row,self.out_col,self.channels)
		self.activation=echo

	def forward(self,inp,training=True):
		self.input_shape=inp.shape
		return inp.repeat(2,axis=2).repeat(2,axis=1)

	def backprop(self,grads,layer=1):
		#grads[self.batches,esz,esz,self.channels],inp[self.batches,row,col,self.channels],kernels[self.ksz,self.ksz],stride[row,col]
		grads=grads.reshape(self.input_shape[0],self.row,self.ksz,self.col,self.ksz,self.channels)
		return grads.sum(axis=(2,4),keepdims=True).reshape(self.input_shape)

class flatten(Layer):
	def __init__(self,name=None):
		super().__init__()
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

	def backprop(self,grads,layer=1):
		return grads.reshape(-1,self.r,self.c,self.channels)

class reshape(Layer):
	def __init__(self,target_shape,name=None):
		super().__init__()
		self.type=self.__class__.__name__
		self.dtype=np.float32
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		self.input_shape=seq_instance.get_inp_shape()
		self.target_shape=target_shape
		tt=1
		for i in self.input_shape:
			tt*=i
		for i in target_shape:
			tt/=i
		if tt!=1:
			raise Exception("Cannot reshape input "+str(self.input_shape)+" to "+str(target_shape)+'.')
		self.shape=(None,*target_shape)
		self.param=0
		self.activation=echo

	def forward(self,inp,training=True):
		return inp.reshape(-1,*self.target_shape)

	def backprop(self,grads,layer=1):
		return grads.reshape(-1,*self.input_shape)

class dense(Layer):
	def __init__(self,num_out,input_shape=None,weights=None,biases=None,activation=echo,mean=0,std=0.01,name=None):
		super().__init__()
		self.dtype=np.float32
		self.type=self.__class__.__name__
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		if input_shape is None:
			self.input_shape=seq_instance.get_inp_shape()[0]
		else:
			self.input_shape=input_shape
		self.activation=activation
		if weights is None:
			self.weights = std*np.random.randn(self.input_shape,num_out).astype(self.dtype) + mean
			# weights/=np.sqrt(self.input_shape)
		else:
			if weights.shape!=(self.input_shape,num_out):
				raise Exception("weights should be of shape: "+str((self.input_shape,num_out)))
			else:
				self.weights = weights
		if biases is None:
			self.biases = std*np.random.randn(1,num_out).astype(self.dtype) + mean
		else:
			if biases.shape!=(1,num_out):
				raise Exception("biases should be of shape: "+str((1,num_out)))
			else:
				self.biases = biases
		self.kernels = self.weights
		self.w_m=0
		self.w_v=0
		self.b_m=0
		self.b_v=0
		self.shape=(None,num_out)
		self.param=self.input_shape*num_out + num_out
		self.not_softmax_cross_entrp=True
		if self.activation==echo:
			self.notEcho=False
		else:
			self.notEcho=True

	def forward(self,inp,training=True):
		self.inp=inp
		self.z_out=np.dot(inp,self.weights)+self.biases
		self.a_out=self.activation(self.z_out)
		return self.a_out

	def backprop(self,grads,layer=1):
		if self.notEcho and self.not_softmax_cross_entrp:			# make it better in future
			grads*=self.activation(self.z_out,self.a_out,derivative=True)
		d_c_b=grads
		self.d_c_w=np.dot(self.inp.T,d_c_b)#/self.inp.shape[0]
		if layer:
			d_c_a=np.dot(d_c_b,self.weights.T)
		else:
			d_c_a=0
		self.d_c_b=d_c_b.sum(axis=0,keepdims=True)
		# self.d_c_b=d_c_b.mean(axis=0,keepdims=True)
		return d_c_a

class dropout(Layer):
	def __init__(self,rate=0.2,name=None):
		super().__init__()
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

	def backprop(self,grads,layer=1):
		return grads*self.mask

class BatchNormalization(Layer):					#Have to add references to each brah
	def __init__(self,momentum=0.9,epsilon=1e-10,name=None):
		super().__init__()
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

	def backprop(self,grads,layer=1):
		#grads(batches,row,col,channels), xmu(batches,row,col,channels)=inp-mean 		#FU
		batches=self.inp_shape[0]
		if batches!=self.batches:
			self.batches=batches
		self.d_c_b=grads.sum(axis=0) 				#(row,col,channels)		# biases is beta
		self.d_c_w=(self.xnorm*grads).sum(axis=0)	#(row,col,channels)		# gamma is weights
		d_inp=(1/self.batches)*self.istd*self.weights*(self.batches*grads-self.d_c_b-self.xmu*self.ivar*((grads*self.xmu).sum(axis=0)))
		return d_inp

class Activation(Layer):
	def __init__(self,activation=echo,input_shape=None,name=None):
		super().__init__()
		self.dtype=np.float32
		self.type=self.__class__.__name__
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		if input_shape is None:
			input_shape=seq_instance.get_inp_shape()
		self.activation=activation
		self.shape=(None,*input_shape)
		self.param=0
		self.not_softmax_cross_entrp=True
		if self.activation==echo:
			self.notEcho=False
		else:
			self.notEcho=True

	def forward(self,inp,training=True):
		self.z_out=inp
		self.a_out=self.activation(self.z_out)
		return self.a_out

	def backprop(self,grads,layer=1):
		if self.notEcho and self.not_softmax_cross_entrp:
			grads*=self.activation(self.z_out,self.a_out,derivative=True)
		return grads

class InputLayer(Layer):		#just placeholder
	def __init__(self,shape=None):
		super().__init__()
		self.name='input_layer'
		self.type=self.__class__.__name__
		self.dtype=np.float32
		try:
			self.shape=(None,*shape)
		except:
			self.shape=(None,shape)
		self.param=0
		self.activation=echo