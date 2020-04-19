#!/usr/bin/env python3
from .Layer import *

def init_kernel_bias(num_inp_channels, kernel_size, num_kernels,mean=0,std=0.01,dtype=cp.float32):
		weights = std*cp.random.randn(num_inp_channels, kernel_size, kernel_size, num_kernels) + mean
		# weights/=cp.sqrt(num_inp_channels)
		bias = std*cp.random.randn(1,num_kernels) + mean
		return weights.astype(dtype,copy=False), bias.astype(dtype,copy=False)

class conv2d(Layer):
	def __init__(self,num_kernels=0,input_shape=None,kernel_size=0,kernels=None,activation=echo,biases=0,stride=(1,1),dilation=(1,1),dlate=(1,1),padding=None,batches=1,backp=True,std=0.01,name=None,out_row=None,out_col=None,off_transpose=0):		#padding=(ksz-1)/2 for same shape in stride 1
		#input_shape[row,col,channels], kernels(channels,ksz,ksz,num_kernels), biases[1,num_ker], stride[row,col]
		super().__init__()
		if input_shape is None:
			input_shape=seq_instance.get_inp_shape()
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
		if self.kernels is None:
			self.kernel_size=kernel_size
			self.num_kernels=num_kernels
			self.kernels,self.biases = init_kernel_bias(self.channels,self.kernel_size,self.num_kernels,std=std,dtype=self.dtype)
		else:
			self.kernel_size=kernels.shape[1]
			self.num_kernels=kernels.shape[3]
		self.w_m=cp.zeros_like(self.weights)
		self.w_v=cp.zeros_like(self.weights)
		self.bias_is_not_0=True
		if cp.isscalar(self.biases):
			if self.biases==0:
				self.bias_is_not_0=False
		if self.bias_is_not_0:
			self.b_m=cp.zeros_like(self.biases)
			self.b_v=cp.zeros_like(self.biases)
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
			self.out_col=(self.ecol-self.kernel_size+2*self.padding-(self.kernel_size-1)*(self.dilation[1]-1))//stride[1]+1
		else:
			self.out_col=out_col
		self.param=(self.kernel_size*self.kernel_size*self.channels+1)*self.num_kernels
		self.off_transpose=off_transpose
		if (self.stride[0]+self.stride[1])>2:
			if backp:
				if self.off_transpose==0:
					cuut=self.padding-self.padding//self.stride[0]
					self.off_transpose=(self.row+2*self.padding)*cuut+cuut
		self.shape=(None,self.out_row,self.out_col,self.num_kernels)
		self.is_not_dker=True
		if backp:
			self.init_back()

	def init_back(self):				# flipped kernel has same reference as original one so it will be updated automatically with original kernel
		self.flipped=self.kernels[:,::-1,::-1,:].transpose(3,1,2,0)	#flipped[num_kernels,kernel_size,kernel_size,channels]
		if (self.stride[0]+self.stride[1])>2:
			padk=self.padding
			padi=self.kernel_size-1
			distride=(1,1)
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
		grads=cp.empty((self.batches,self.out_row,self.out_col,self.num_kernels),dtype=self.dtype)
		self.d_ker=conv2d(input_shape=(self.row,self.col,self.batches),kernels=grads,activation=echo,dilation=self.stride,dlate=self.dlate,padding=padk,backp=False,off_transpose=off_transpose_ker,out_row=self.kernel_size,out_col=self.kernel_size,batches=self.channels)
		self.d_ker.is_not_dker=False
		self.d_inp=conv2d(input_shape=(self.out_row,self.out_col,self.num_kernels),kernels=self.flipped,activation=echo,stride=distride,dlate=self.stride,padding=padi,off_transpose=off_transpose_inp,backp=False,out_row=self.row,out_col=self.col)

	def forward(self,inp,training=True):
		self.inp=inp.transpose(0,3,1,2)
		#inp[batches,channels,row,col]
		batches,channels=self.inp.shape[:2]
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
		self.d_ker.padded=cp.ascontiguousarray(self.padded.transpose(1,0,2,3))
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
