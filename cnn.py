#!/usr/bin/env python3
import numpy as np

sd=470#np.random.randint(1000)
# print(sd)
np.random.seed(sd)	#470

class conv_net:
	def __init__(self):
		self.learning_rate=0.01

	def init_kernel_bias(self, num_inp_channels, kernel_size, num_kernels):
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

	def conv2d(self,inp,kernels,biases,stride=[1,1],padding=1):
		#inp[batches,row,col,d],kernels(d,ksz,ksz,num_ker),biases[1,num_ker],stride[row,col]
		inp=inp.transpose(0,3,1,2)	#inp[batches,d,row,col]
		output=[]
		ksz=kernels.shape[1]
		out_row,out_col=((inp.shape[2]-ksz+2*padding)//stride[0]+1),((inp.shape[3]-ksz+2*padding)//stride[1]+1)
		for img in inp:		#img[d,row,col]
			padded=np.zeros((img.shape[0],img.shape[1]+2*padding,img.shape[2]+2*padding))
			padded[:,padding:-padding,padding:-padding]=img
			# Take all windows into a matrix
			d,row,col=padded.shape
			window=(np.arange(ksz)[:,None]*row+np.arange(ksz)).ravel()+np.arange(d)[:,None]*row*col
			slider=(np.arange(out_row)[:,None]*row+np.arange(out_col))
			# windows(out_row*out_col, ksz*ksz*d) . kernels(d*ksz*ksz,num_ker)
			out=(np.dot(np.take(padded, window.ravel()+slider[::stride[0],::stride[1]].ravel()[:,None]), kernels.reshape(-1,kernels.shape[3])))
			out=(out+biases).reshape(out_row,out_col,kernels.shape[3])
			output.append(out)
		return np.array(output)

	def conv2d_back(self,errors,inp,kernels,biases,stride=[1,1],padding=1):
		#errors[esz,esz,num_ker],inp[batches,row,col,d],kernels(d,ksz,ksz,num_ker),biases[1,num_ker],stride[row,col]
		esz,esz,num_ker=errors.shape
		errors=errors.reshape(-1,num_ker).repeat(inp.shape[3],axis=0).reshape(esz,esz,inp.shape[3],num_ker)
		errors=errors.transpose(2,0,1,3)	#errors[d,esz,esz,num_ker]
		inp=inp.transpose(0,3,1,2)	#inp[batches,d,row,col]
		flipped=np.flip(kernels,(1,2)).transpose(0,3,1,2)	#flipped[d,num_ker,ksz,ksz]
		ksz=flipped.shape[2]
		d_kernels=[]
		d_inputs=[]
		out_row,out_col=((inp.shape[2]-esz+2*padding)//stride[0]+1),((inp.shape[3]-esz+2*padding)//stride[1]+1)
		_,d,er_r,er_c=inp.shape
		for img in inp:		#img[d,row,col]
			# Backprop for kernels.
			padded=np.zeros((img.shape[0],img.shape[1]+2*padding,img.shape[2]+2*padding))
			padded[:,padding:-padding,padding:-padding]=img
			# Take all windows into a matrix
			d,row,col=padded.shape
			window=(np.arange(esz)[:,None]*row+np.arange(esz)).ravel()+np.arange(d)[:,None]*row*col
			slider=(np.arange(out_row)[:,None]*row+np.arange(out_col))
			# windows(out_row*out_col, esz*esz*d) . errors(d*esz*esz,num_ker)
			d_ker=(np.dot(np.take(padded, window.ravel()+slider[::stride[0],::stride[1]].ravel()[:,None]), errors.reshape(-1,num_ker)))
			d_ker=d_ker.reshape(out_row,out_col,d,num_ker)
			d_kernels.append(d_ker)

			# Backprop for inp.		errors[d,esz,esz,num_ker]
			errors=errors.transpose(0,3,1,2)	#errors[d,num_kernel,esz,esz]
			d_inp=[]
			for channel,flip in zip(errors,flipped):	#channel[num_ker,esz,esz], flip[num_ker,ksz,ksz]
				padded=np.zeros((num_ker,esz+2*padding,esz+2*padding))
				padded[:,padding:-padding,padding:-padding]=channel
				# Take all windows into a matrix
				num_ker,row,col=padded.shape
				window=(np.arange(ksz)[:,None]*row+np.arange(ksz)).ravel()+np.arange(num_ker)[:,None]*row*col
				slider=(np.arange(er_r)[:,None]*row+np.arange(er_c))
				# windows(er_r*er_c, ksz*ksz*num_ker) . flip(num_ker*ksz*ksz,1)
				d_chnl=(np.dot(np.take(padded, window.ravel()+slider[::stride[0],::stride[1]].ravel()[:,None]), flip.reshape(-1,1)))
				d_chnl=d_chnl.reshape(er_r,er_c)
				d_inp.append(d_chnl)
			# d_inp[d,row,col]
			d_inp=np.array(d_inp).transpose(1,2,0)
			# d_inp[row,col,d]
			d_inputs.append(d_inp)

		return np.array(d_kernels)*learning_rate,np.array(d_inputs)*learning_rate