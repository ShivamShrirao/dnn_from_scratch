#!/usr/bin/env python3
from .Layer import *

class conv2d(Layer):
	def __init__(self,num_kernels=0,input_shape=None,kernel_size=0,kernels=None,activation=echo,biases=None,stride=[1,1],dilation=[1,1],dlate=[1,1],padding=None,batches=1,backp=True,std=0.01,name=None,out_row=None,out_col=None,off_transpose=0):		#padding=(ksz-1)/2 for same shape in stride 1
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
		self.w_m=cp.zeros_like(self.weights)
		self.w_v=cp.zeros_like(self.weights)
		self.b_m=cp.zeros_like(self.biases)
		self.b_v=cp.zeros_like(self.biases)
		if self.kernels is None:
			self.kernel_size=kernel_size
			self.num_kernels=num_kernels
			self.kernels,self.biases = self.init_kernel_bias(self.channels,self.kernel_size,self.num_kernels,std=std)
		else:
			self.kernel_size=kernels.shape[1]
			self.num_kernels=kernels.shape[3]
		if biases != None:
			self.biases=biases
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
