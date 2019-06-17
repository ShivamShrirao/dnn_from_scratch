#!/usr/bin/env python3
import numpy as np

sd=np.random.randint(1000)
# print(sd)
np.random.seed(sd)	#470
learning_rate=0.01
seq_instance=None

class conv2d:
	def __init__(self,num_kernels=0,input_shape=None,kernel_size=0,kernels=None,biases=0,stride=[1,1],padding=0,backp=True,name=None):		#padding=(ksz-1)/2 for same shape in stride 1
		#input_shape[row,col,d],kernels(d,ksz,ksz,num_kernels),biases[1,num_ker],stride[row,col]
		if input_shape is None:
			input_shape=seq_instance.get_inp_shape()
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		self.input_shape=input_shape
		self.row,self.col,self.d=input_shape
		self.batches=1
		self.kernels=kernels
		if self.kernels is None:
			self.kernel_size=kernel_size
			self.num_kernels=num_kernels
			self.kernels,self.biases = self.init_kernel_bias(self.d,self.kernel_size,self.num_kernels)
		else:
			self.kernel_size=kernels.shape[1]
			self.num_kernels=kernels.shape[3]
		self.kern = self.kernels.reshape(-1,self.num_kernels)
		self.padding=padding
		if not self.padding:							#take care of padding in backprop too
			self.padding=(self.kernel_size-1)//2					#currently don't give 'even' self.kernel_size
		self.out_row,self.out_col=((self.row-self.kernel_size+2*self.padding)//stride[0]+1),((self.col-self.kernel_size+2*self.padding)//stride[1]+1)
		self.row+=2*self.padding
		self.col+=2*self.padding
		self.padded=np.zeros((self.batches,self.d,self.row,self.col))
		# Take all windows into a matrix
		window=(np.arange(self.kernel_size)[:,None]*self.row+np.arange(self.kernel_size)).ravel()+np.arange(self.d)[:,None]*self.row*self.col
		slider=(np.arange(self.out_row*stride[0])[:,None]*self.row+np.arange(self.out_col*stride[1]))
		self.ind = window.ravel()+slider[::stride[0],::stride[1]].ravel()[:,None]
		self.output=np.empty((self.batches,self.out_row*self.out_col,self.num_kernels))
		# bind= np.arange(self.batches)[:,None]*self.d*self.row*self.col+self.ind.ravel()		#for self.batches
		self.shape=(None,self.out_row,self.out_col,self.num_kernels)
		if backp:
			self.init_back()
	def init_back(self):
		self.flipped=self.kernels[:,::-1,::-1,:].transpose(3,1,2,0)	#self.flipped[num_kernels,self.kernel_size,self.kernel_size,d]
		pad=(self.kernel_size-1)//2
		errors=self.output.reshape(self.batches,self.out_row,self.out_col,self.num_kernels)
		self.d_ker=conv2d(input_shape=self.input_shape,kernels=errors,padding=pad,backp=False)
		self.d_inp=conv2d(input_shape=(self.out_row,self.out_col,self.num_kernels),kernels=self.flipped,backp=False)
	def init_kernel_bias(self,num_inp_channels, kernel_size, num_kernels,mean=0,std=0.1):
		shape = [num_inp_channels, kernel_size, kernel_size, num_kernels]
		weights = std*np.random.randn(*shape) + mean
		# weights/=np.sqrt(kernel_size*kernel_size*num_inp_channels)
		bias = std*np.random.randn(1,num_kernels) + mean
		return weights, bias

	def forward(self,inp):
		self.inp=inp.transpose(0,3,1,2)  #inp[batches,self.d,self.row,self.col]
		batches=self.inp.shape[0]
		if self.batches!=batches:
			self.batches=batches
			self.padded=np.zeros((self.batches,self.d,self.row,self.col))
			self.output=np.empty((self.batches,self.out_row*self.out_col,self.num_kernels))
		self.padded[:,:,self.padding:-self.padding,self.padding:-self.padding]=self.inp
		for i,img in enumerate(self.padded):		#img[self.d,self.row,self.col]
			# windows(self.out_row*self.out_col, kernel_size*kernel_size*self.d) . kernels(self.d*kernel_size*kernel_size,self.num_kernels)
			self.output[i]=np.dot(np.take(img, self.ind), self.kern)+self.biases
		# output=np.array([(np.dot(np.take(i,self.ind),self.kern)+self.biases) for i in padded]).reshape(self.batches,self.out_row,self.out_col,self.num_kernels)
		# output=(np.dot(np.take(padded, bind).reshape(-1,self.d*kernel_size*kernel_size), self.kern)+self.biases)
					# [self.batches*self.out_row*self.out_col,self.d*kernel_size*kernel_size] . [self.d*kernel_size*kernel_size, self.num_kernels]
		return self.output.reshape(self.batches,self.out_row,self.out_col,self.num_kernels)

	def backprop(self,errors,layer=1):								#strides[batch,row,col,depth]
		#errors[batches,esz,esz,num_kernels],inp[batches,row,col,d],kernels(d,kernel_size,kernel_size,num_kernels),biases[1,num_kernels],stride[row,col]
		self.d_ker.kernels=errors
		d_kernels=self.d_ker.forward(self.inp)
		d_kernels/=self.batches		#take mean change over batches
		# Backprop for inp.		errors[batches,esz,esz,num_kernels]	self.flipped[num_kernels,kernel_size,kernel_size,d]
		if layer:
			d_inputs=self.d_inp.forward(errors)
		else:
			d_inputs=0
		d_bias=self.d_ker.kern.mean(axis=0,keepdims=True)

		return d_inputs, d_kernels, d_bias

class max_pool:
	def __init__(self,input_shape=None,ksize=[2,2],stride=[2,2],name=None):
		#inp[batches,row,col,d], kernels[ksz,ksz], stride[row,col]
		self.ksz=ksize[0]
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		if input_shape is None:
			input_shape=seq_instance.get_inp_shape()
		row,col,self.d=input_shape
		self.out_row,self.out_col=row//self.ksz,col//self.ksz
		self.shape=(None,self.out_row,self.out_col,self.d)
	def forward(self,inp):
		self.input_shape=inp.shape
		self.batches=self.input_shape[0]
		ipp=inp.reshape(self.batches,self.out_row,self.ksz,self.out_col,self.ksz,self.d)
		output=ipp.max(axis=(2,4),keepdims=True)
		self.mask=(ipp==output)
		return output.reshape(self.batches,self.out_row,self.out_col,self.d)

	def backprop(self,errors):
		#errors[self.batches,esz,esz,self.d],inp[self.batches,row,col,self.d],kernels[self.ksz,self.ksz],stride[row,col]
		return (self.mask*errors.reshape(self.batches,self.out_row,1,self.out_col,1,self.d)).reshape(self.input_shape)

class flatten:
	def __init__(self,name=None):
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		input_shape=seq_instance.get_inp_shape()
		self.r,self.c,self.d=input_shape
		self.fsz=self.r*self.c*self.d
		self.shape=(None,self.fsz)

	def forward(self,inp):
		return inp.reshape(-1,self.fsz)

	def backprop(self,errors):
		return errors.reshape(-1,self.r,self.c,self.d)

class dense:
	def __init__(self,num_out,num_inp=None,mean=0,std=0.1,name=None):
		if name is None:
			self.name=self.__class__.__name__
		else:
			self.name=name
		if num_inp is None:
			num_inp=seq_instance.get_inp_shape()[0]
		self.weights = std*np.random.randn(num_inp,num_out) + mean
		# weights/=np.sqrt(num_inp)
		self.biases = std*np.random.randn(1,num_out) + mean
		self.shape=(None,num_out)

	def forward(self,inp):
		self.inp=inp
		return np.dot(inp,self.weights)+self.biases

	def backprop(self,errors,layer=1):
		d_c_b=errors
		d_c_w=np.dot(self.inp.T,d_c_b)/self.inp.shape[0]
		if layer:
			d_c_a=np.dot(d_c_b,self.weights.T)
		else:
			d_c_a=0
		return d_c_a, d_c_w, d_c_b.mean(axis=0,keepdims=True)