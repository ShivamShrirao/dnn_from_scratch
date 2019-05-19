#!/usr/bin/env python3
import numpy as np

sd=470#np.random.randint(1000)
# print(sd)
np.random.seed(sd)	#470

class conv_net:
	# def __init__(self):

	def init_kernel(self, kernel_size, num_inp_channels, num_kernels):
		shape = [num_inp_channels, kernel_size, kernel_size, num_kernels]
		weights = 0.1*np.random.randn(*shape)
		bias = 0.2*np.random.randn(num_kernels,1)
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

	def conv2d(self,inp,kernels,biases,strides=[1,1,1,1],padding=1):
		#inp[num,x,y,d],kernels(d,ksz,ksz,num),biases[num,1],strides[batch,x,y,d]
		output=[]
		ksz=kernels.shape[1]
		out_x,out_y=((inp.shape[1]-ksz+2*padding)//strides[1]+1),((inp.shape[2]-ksz+2*padding)//strides[2]+1)
		inp=inp.transpose(0,3,1,2)
		for img in inp:		#img[d,x,y]
			padded=np.zeros((img.shape[0],img.shape[1]+2*padding,img.shape[2]+2*padding))
			padded[:,padding:-padding,padding:-padding]=img
			out=[]
			for j in range(0,out_y,strides[2]):
				for i in range(0,out_x,strides[1]):
					out.append(np.dot(padded[:,i:i+ksz,j:j+ksz].reshape(1,-1),kernels.reshape(-1,kernels.shape[3]))[0])
			output.append(np.array(out).reshape(out_x,out_y,kernels.shape[3]))
		return np.array(output)
	# def backprop(self, y):
		