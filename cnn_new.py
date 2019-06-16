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

	def conv2d(self,inp,kernels,biases,stride=[1,1],padding=0):		#padding=(ksz-1)/2 for same shape in stride 1
		#inp[batches,row,col,d],kernels(d,ksz,ksz,num_ker),biases[1,num_ker],stride[row,col]
		inp=inp.transpose(0,3,1,2)  #inp[batches,d,row,col]
		ksz=kernels.shape[1]
		num_ker=kernels.shape[3]
		if not padding:							#take care of padding in backprop too
			padding=(ksz-1)//2					#currently don't give 'even' ksz
		out_row,out_col=((inp.shape[2]-ksz+2*padding)//stride[0]+1),((inp.shape[3]-ksz+2*padding)//stride[1]+1)
		batches,d,row,col=inp.shape
		row+=2*padding
		col+=2*padding
		padded=np.zeros((batches,d,row,col))
		padded[:,:,padding:-padding,padding:-padding]=inp
		# Take all windows into a matrix
		kern = kernels.reshape(-1,num_ker)
		window=(np.arange(ksz)[:,None]*row+np.arange(ksz)).ravel()+np.arange(d)[:,None]*row*col
		slider=(np.arange(out_row*stride[0])[:,None]*row+np.arange(out_col*stride[1]))
		ind = window.ravel()+slider[::stride[0],::stride[1]].ravel()[:,None]
		output=np.empty((batches,out_row*out_col,num_ker))
		for i,img in enumerate(padded):		#img[d,row,col]
			# windows(out_row*out_col, ksz*ksz*d) . kernels(d*ksz*ksz,num_ker)
			output[i]=np.dot(np.take(img, ind), kern)+biases
		# output=np.array([(np.dot(np.take(i,ind),kern)+biases) for i in padded]).reshape(batches,out_row,out_col,num_ker)
		# bind= np.arange(batches)[:,None]*d*row*col+ind.ravel()		#for batches
		# output=(np.dot(np.take(padded, bind).reshape(-1,d*ksz*ksz), kern)+biases)
					# [batches*out_row*out_col,d*ksz*ksz] . [d*ksz*ksz, num_ker]
		return output.reshape(batches,out_row,out_col,num_ker)

	def conv2d_back(self,errors,inp,kernels,biases,stride=[1,1],layer=1):								#strides[batch,row,col,depth]
		#errors[batches,esz,esz,num_ker],inp[batches,row,col,d],kernels(d,ksz,ksz,num_ker),biases[1,num_ker],stride[row,col]
		batches,esz,esz,num_ker=errors.shape
		inp=inp.transpose(3,1,2,0)		#inp[d,row,col,batches]
		flipped=np.flip(kernels,(1,2)).transpose(3,1,2,0)	#flipped[num_ker,ksz,ksz,d]
		ksz=flipped.shape[1]
		pad=(ksz-1)//2
		d_kernels=self.conv2d(inp,errors,0,padding=pad)
		d_kernels/=batches		#take mean change over batches
		# Backprop for inp.		errors[batches,esz,esz,num_ker]	flipped[num_ker,ksz,ksz,d]
		if layer:
			d_inputs=self.conv2d(errors,flipped,0)
		else:
			d_inputs=0
		d_bias=errors.reshape(-1,num_ker).mean(axis=0,keepdims=True)

		return d_inputs, d_kernels*self.learning_rate, d_bias*self.learning_rate

	def max_pool(self,inp,ksize=[2,2],stride=[2,2]):
		#inp[batches,row,col,d], kernels[ksz,ksz], stride[row,col]
		ksz=ksize[0]
		batches,row,col,d=inp.shape
		out_row,out_col=row//ksz,col//ksz
		ipp=inp.reshape(batches,out_row,ksz,out_col,ksz,d)
		output=ipp.max(axis=(2,4),keepdims=True)
		mask=((ipp-output)==0)
		#[batches,o_row,o_col,d]
		return output.squeeze().reshape(batches,out_row,out_col,d), mask

	def max_pool_back(self,errors,inp,mask,ksize=[2,2],stride=[2,2]):
		#errors[batches,esz,esz,d],inp[batches,row,col,d],kernels[ksz,ksz],stride[row,col]
		ksz=ksize[0]
		batches,row,col,d=inp.shape
		out_row,out_col=row//ksz,col//ksz
		return (mask*errors.reshape(batches,out_row,1,out_col,1,d)).reshape(inp.shape)