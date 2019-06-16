#!/usr/bin/env python3
import numpy as np

sd=np.random.randint(1000)
# print(sd)
np.random.seed(sd)	#470

class conv_net:
	def __init__(self):
		self.learning_rate=0.01

	def init_kernel_bias(self, num_inp_channels, kernel_size, num_kernels, std=0.1):
		shape = [num_inp_channels, kernel_size, kernel_size, num_kernels]
		weights = std*np.random.randn(*shape)
		# weights/=np.sqrt(kernel_size*kernel_size*num_inp_channels)
		bias = std*np.random.randn(1,num_kernels)
		return weights, bias

	def __str__(self):
		return str(self.__dict__)

	def sigmoid(self,x):
		x=np.clip(x,-500,500)
		return 1.0/(1+np.exp(-x))

	def sigmoid_der(self,x,y):
		return x * (1 - x)

	def elliot_function(signal,derivative=False):
		""" A fast approximation of sigmoid """
		s = 1 # steepness
		
		abs_signal = (1 + np.abs(signal * s))
		if derivative:
			return 0.5 * s / abs_signal**2
		else:
			# Return the activation signal
			return 0.5*(signal * s) / abs_signal + 0.5

	def relu(self,x):
		x[x<0]=0
		return x

	def relu_der(self,x,y):
		return (y > 0)

	def softmax(self,x):
		# exps = np.exp(x)
		exps = np.exp(x-np.max(x, axis=1, keepdims = True))
		return exps/np.sum(exps, axis=1, keepdims = True)

	def soft_der(self,x,y):
		return np.ones(self.softmax(x).shape)

	def del_cross_soft(self,out,res):
		res = res.argmax(axis=1)
		m = res.shape[0]
		grad = out
		grad[range(m),res]-=1
		grad = grad/m
		return grad

	def normalize(self,x):
		mn=x.min()
		mx=x.max()
		x = (x-mn)/(mx-mn)
		return x

	def batch_norm(self,aa):
		gamma=aa.std()
		beta=aa.mean()
		ad=(aa-beta)/gamma				# normalize
		ad=ad*gamma+beta				# recover
		return ad

class conv2d:
	def __init__(self,inp,kernels,biases,stride=[1,1],padding=0,backp=True):		#padding=(ksz-1)/2 for same shape in stride 1
		#inp[batches,row,col,d],kernels(d,ksz,ksz,num_ker),biases[1,num_ker],stride[row,col]
		inp=inp.transpose(0,3,1,2)  #inp[batches,d,row,col]
		ksz=kernels.shape[1]
		self.num_ker=kernels.shape[3]
		self.padding=padding
		if not self.padding:							#take care of padding in backprop too
			self.padding=(ksz-1)//2					#currently don't give 'even' ksz
		self.out_row,self.out_col=((inp.shape[2]-ksz+2*self.padding)//stride[0]+1),((inp.shape[3]-ksz+2*self.padding)//stride[1]+1)
		self.batches,self.d,self.row,self.col=inp.shape
		self.row+=2*self.padding
		self.col+=2*self.padding
		self.padded=np.zeros((self.batches,self.d,self.row,self.col))
		# Take all windows into a matrix
		self.init_kern(kernels)
		self.biases = biases
		window=(np.arange(ksz)[:,None]*self.row+np.arange(ksz)).ravel()+np.arange(self.d)[:,None]*self.row*self.col
		slider=(np.arange(self.out_row*stride[0])[:,None]*self.row+np.arange(self.out_col*stride[1]))
		self.ind = window.ravel()+slider[::stride[0],::stride[1]].ravel()[:,None]
		self.output=np.empty((self.batches,self.out_row*self.out_col,self.num_ker))
		# bind= np.arange(self.batches)[:,None]*self.d*self.row*self.col+self.ind.ravel()		#for self.batches
		if backp:
			self.init_back()
	def init_kern(self,kernels):
		self.kern = kernels.reshape(-1,self.num_ker)
	def init_back(self):
		self.flipped=np.flip(kernels,(1,2)).transpose(3,1,2,0)	#self.flipped[num_ker,ksz,ksz,d]
		errors=self.output.reshape(self.batches,self.out_row,self.out_col,self.num_ker)
		ksz=self.flipped.shape[1]
		pad=(ksz-1)//2
		self.d_ker=conv2d(inp,errors,0,padding=pad,backp=False)
		self.d_inp=conv2d(errors,self.flipped,0,backp=False)

	def forward(self,inp):
		self.inp=inp.transpose(0,3,1,2)  #inp[batches,self.d,self.row,self.col]
		batches=self.inp.shape[0]
		if self.batches!=batches:
			self.batches=batches
			self.padded=np.zeros((self.batches,self.d,self.row,self.col))
			self.output=np.empty((self.batches,self.out_row*self.out_col,self.num_ker))
		self.padded[:,:,self.padding:-self.padding,self.padding:-self.padding]=self.inp
		for i,img in enumerate(self.padded):		#img[self.d,self.row,self.col]
			# windows(self.out_row*self.out_col, ksz*ksz*self.d) . kernels(self.d*ksz*ksz,self.num_ker)
			self.output[i]=np.dot(np.take(img, self.ind), self.kern)+self.biases
		# output=np.array([(np.dot(np.take(i,self.ind),self.kern)+self.biases) for i in padded]).reshape(self.batches,self.out_row,self.out_col,self.num_ker)
		# output=(np.dot(np.take(padded, bind).reshape(-1,self.d*ksz*ksz), self.kern)+self.biases)
					# [self.batches*self.out_row*self.out_col,self.d*ksz*ksz] . [self.d*ksz*ksz, self.num_ker]
		return self.output.reshape(self.batches,self.out_row,self.out_col,self.num_ker)

	def backprop(self,errors,layer=1):								#strides[batch,row,col,depth]
		#errors[batches,esz,esz,num_ker],inp[batches,row,col,d],kernels(d,ksz,ksz,num_ker),biases[1,num_ker],stride[row,col]
		self.d_ker.init_kern(errors)
		d_kernels=self.d_ker.forward(self.inp)
		d_kernels/=self.batches		#take mean change over batches
		# Backprop for inp.		errors[batches,esz,esz,num_ker]	self.flipped[num_ker,ksz,ksz,d]
		if layer:
			d_inputs=self.d_inp.forward(errors)
		else:
			d_inputs=0
		d_bias=self.d_ker.kern.mean(axis=0,keepdims=True)

		return d_inputs, d_kernels*self.learning_rate, d_bias*self.learning_rate

class max_pool:
	def __init__(self,inp,ksize=[2,2],stride=[2,2]):
		#inp[batches,row,col,d], kernels[ksz,ksz], stride[row,col]
		self.ksz=ksize[0]
		self.batches,row,col,self.d=inp.shape
		self.out_row,self.out_col=row//self.ksz,col//self.ksz
	def forward(self,inp):
		self.inp_shape=inp.shape
		self.batches=self.inp_shape[0]
		ipp=inp.reshape(self.batches,self.out_row,self.ksz,self.out_col,self.ksz,self.d)
		output=ipp.max(axis=(2,4),keepdims=True)
		self.mask=(ipp==output)
		#[self.batches,o_row,o_col,self.d]
		return output.squeeze().reshape(self.batches,self.out_row,self.out_col,self.d)

	def backprop(self,errors):
		#errors[self.batches,esz,esz,self.d],inp[self.batches,row,col,self.d],kernels[self.ksz,self.ksz],stride[row,col]
		return (self.mask*errors.reshape(self.batches,self.out_row,1,self.out_col,1,self.d)).reshape(self.inp_shape)