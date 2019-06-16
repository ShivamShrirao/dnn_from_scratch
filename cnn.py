#!/usr/bin/env python3
import numpy as np

sd=np.random.randint(1000)
# print(sd)
np.random.seed(sd)	#470
learning_rate=0.01

def init_kernel_bias(num_inp_channels, kernel_size, num_kernels,mean=0,std=0.1):
	shape = [num_inp_channels, kernel_size, kernel_size, num_kernels]
	weights = std*np.random.randn(*shape) + mean
	# weights/=np.sqrt(kernel_size*kernel_size*num_inp_channels)
	bias = std*np.random.randn(1,num_kernels) + mean
	return weights, bias

class conv2d:
	def __init__(self,inp,kernels,biases=0,stride=[1,1],padding=0,backp=True):		#padding=(ksz-1)/2 for same shape in stride 1
		#inp[batches,row,col,d],kernels(d,ksz,ksz,num_ker),biases[1,num_ker],stride[row,col]
		self.inp=inp.transpose(0,3,1,2)  #inp[batches,d,row,col]
		self.kernels = kernels
		self.biases = biases
		self.ksz=self.kernels.shape[1]
		self.num_ker=self.kernels.shape[3]
		self.kern = self.kernels.reshape(-1,self.num_ker)
		self.padding=padding
		if not self.padding:							#take care of padding in backprop too
			self.padding=(self.ksz-1)//2					#currently don't give 'even' self.ksz
		self.out_row,self.out_col=((inp.shape[2]-self.ksz+2*self.padding)//stride[0]+1),((inp.shape[3]-self.ksz+2*self.padding)//stride[1]+1)
		self.batches,self.d,self.row,self.col=inp.shape
		self.row+=2*self.padding
		self.col+=2*self.padding
		self.padded=np.zeros((self.batches,self.d,self.row,self.col))
		# Take all windows into a matrix
		window=(np.arange(self.ksz)[:,None]*self.row+np.arange(self.ksz)).ravel()+np.arange(self.d)[:,None]*self.row*self.col
		slider=(np.arange(self.out_row*stride[0])[:,None]*self.row+np.arange(self.out_col*stride[1]))
		self.ind = window.ravel()+slider[::stride[0],::stride[1]].ravel()[:,None]
		self.output=np.empty((self.batches,self.out_row*self.out_col,self.num_ker))
		# bind= np.arange(self.batches)[:,None]*self.d*self.row*self.col+self.ind.ravel()		#for self.batches
		self.shape=(None,*self.output.shape[1:])
		if backp:
			self.init_back()
	def init_back(self):
		self.flipped=self.kernels[:,::-1,::-1,:].transpose(3,1,2,0)	#self.flipped[num_ker,self.ksz,self.ksz,d]
		errors=self.output.reshape(self.batches,self.out_row,self.out_col,self.num_ker)
		self.ksz=self.flipped.shape[1]
		pad=(self.ksz-1)//2
		self.d_ker=conv2d(self.inp, errors,padding=pad,backp=False)
		self.d_inp=conv2d(errors, self.flipped,backp=False)

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
		self.d_ker.kernels=errors
		d_kernels=self.d_ker.forward(self.inp)
		d_kernels/=self.batches		#take mean change over batches
		# Backprop for inp.		errors[batches,esz,esz,num_ker]	self.flipped[num_ker,ksz,ksz,d]
		if layer:
			d_inputs=self.d_inp.forward(errors)
		else:
			d_inputs=0
		d_bias=self.d_ker.kern.mean(axis=0,keepdims=True)

		return d_inputs, d_kernels*learning_rate, d_bias*learning_rate

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

class dense:
	def __init__(self,num_inp,num_out,mean=0,std=0.1):
		self.weights = std*np.random.randn(num_inp,num_out) + mean
		# weights/=np.sqrt(num_inp)
		self.biases = std*np.random.randn(1,num_out) + mean

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
		return d_c_a, d_c_w*learning_rate, d_c_b.mean(axis=0,keepdims=True)*learning_rate