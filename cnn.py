#!/usr/bin/env python3
import numpy as np

sd=470#np.random.randint(1000)
# print(sd)
np.random.seed(sd)	#470

class conv_net:
	# def __init__(self):

	def init_kernel(self, num_inp_channels, kernel_size, num_kernels):
		shape = [num_inp_channels, kernel_size, kernel_size, num_kernels]
		weights = 0.1*np.random.randn(*shape)
		bias = 0.2*np.random.randn(1,num_kernels)
		return weights, bias

	def __str__(self):
		return str(self.__dict__)

	def sigmoid(self,x):
		return 1.0/(1+np.exp(-x))

	def sigmoid_der(self,x,y):
		return x * (1 - x)

	def relu(self,x):
		return x*(x>0)

	def relu_der(self,x,y):
		return (y > 0)*1

	def softmax(self,x):
		# exps = np.exp(x)
		exps = np.exp(x-np.max(x))
		return exps/np.sum(exps)

	# def soft_der(self,x,y):
	# 	# return -x*y
	# 	return 1

	# def del_cross_soft(self,out,res):
	# 	res = res.argmax(axis=1)
	# 	m = res.shape[0]
	# 	grad = out
	# 	grad[range(m),res] -= 1
	# 	grad = grad/m
	# 	return grad

	def batch_norm(self,aa):
		gamma=aa.std()
		beta=aa.mean()
		ad=(aa-beta)/gamma				# normalize
		ad=ad*gamma+beta				# recover
		return ad

	def conv2d(self,inp,kernels,biases,strides=[1,1],padding=1):
		#inp[num,row,col,d],kernels(d,ksz,ksz,num_ker),biases[1,num_ker],strides[row,col]
		inp=inp.transpose(0,3,1,2)	#inp[num,d,row,col]
		output=[]
		ksz=kernels.shape[1]
		out_row,out_col=((inp.shape[2]-ksz+2*padding)//strides[0]+1),((inp.shape[3]-ksz+2*padding)//strides[1]+1)
		for img in inp:		#img[d,row,col]
			padded=np.zeros((img.shape[0],img.shape[1]+2*padding,img.shape[2]+2*padding))
			padded[:,padding:-padding,padding:-padding]=img
			# Take all windows into a matrix
			d,row,col=padded.shape
			window=(np.arange(ksz)[:,None]*row+np.arange(ksz)).ravel()+np.arange(d)[:,None]*row*col
			slider=(np.arange(out_row)[:,None]*row+np.arange(out_col))
			# (out_row*out_col, ksz*ksz*d) . (ksz,ksz,depth,num_ker)
			out=(np.dot(np.take(padded, window.ravel()+slider[::strides[0],::strides[1]].ravel()[:,None]), kernels.reshape(-1,kernels.shape[3])))
			out=(out+biases).reshape(out_row,out_col,kernels.shape[3])
			output.append(out)
		return np.array(output)

	def conv2d_back(self,error,inp,kernels,biases):
		pass