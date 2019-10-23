#!/usr/bin/env python3
from nnet import layers
from nnet.functions import *
from nnet.optimizers import *
import pickle
from gc import collect


### TO-DO- In train/fit unifunc, transpose whole data of inp at once and remove from layers.

class Sequential:
	def __init__(self,sequence=[]):
		layers.seq_instance=self
		self.sequence=sequence
		self.learning_rate=0.001

	def add(self,obj):
		self.sequence.append(obj)

	def get_inp_shape(self):
		return self.sequence[-1].shape[1:]

	def predict(self,X_inp):
		self.svd_inp=X_inp[:1]
		for obj in self.sequence:
			X_inp=obj.forward(X_inp,training=False)
		return X_inp

	def fit(self,X_inp,labels):
		for obj in self.sequence:
			X_inp=obj.forward(X_inp)
		err=self.del_loss(X_inp,labels)
		i=self.lenseq_m1
		for obj in self.sequence[::-1]:
			err=obj.backprop(err,layer=i)
			i-=1
		self.optimizer(self.sequence,self.learning_rate,self.beta)
		return X_inp

	def free(self):			#just to free memory of large batch after predict
		X_inp=self.svd_inp
		for obj in self.sequence:
			X_inp=obj.forward(X_inp,training=False)
		err=X_inp
		i=self.lenseq_m1
		for obj in self.sequence[::-1]:
			err=obj.backprop(err,layer=i)
			i-=1
		collect()

	def compile(self,optimizer=adam,beta=0.9,loss=cross_entropy_with_logits,learning_rate=0.001):
		self.optimizer=optimizer
		self.beta=beta
		self.learning_rate=learning_rate
		self.loss=loss
		if self.loss==cross_entropy_with_logits:
			self.sequence[-1].cross_entrp=True
			self.del_loss=del_cross_soft
		elif self.loss==mean_squared_error:
			self.del_loss=del_mean_squared_error
		self.lenseq_m1=len(self.sequence)-1

	def save_weights(self,path):	# has problems if u wanna train the network further. Need to fix that.
		sv_me=[]					# OK for just validation and prediction.
		for obj in self.sequence:
			if obj.param>0:
				if obj.__class__==layers.BatchNormalization:
					sv_me.append((obj.weights,obj.biases,obj.moving_mean,obj.moving_var))
				else:
					sv_me.append((obj.weights,obj.biases))#,obj.w_m,obj.w_v,obj.b_m,obj.b_v))
		with open(path,'wb') as f:
			pickle.dump(sv_me,f)

	def load_weights(self,path):
		with open(path,'rb') as f:
			sv_me=pickle.load(f)
		idx=0
		for obj in self.sequence:
			if obj.param>0:
				if obj.__class__==layers.BatchNormalization:
					obj.weights,obj.biases,obj.moving_mean,obj.moving_var=sv_me[idx]
				else:
					obj.weights,obj.biases=sv_me[idx]
					# obj.weights,obj.biases,obj.w_m,obj.w_v,obj.b_m,obj.b_v=sv_me[idx]
					if obj.__class__==layers.conv2d:
						obj.init_back()
				obj.kernels=obj.weights
				idx+=1

	def summary(self):
		ipl=layers.InputLayer(self.sequence[0].input_shape)
		reps=90
		print(chr(9149)*reps)
		print("Layer (type)".ljust(25)," Output Shape".ljust(25),"Activation".ljust(17),"Param #")
		print('='*reps)
		print('0 {}({})'.format(ipl.name,ipl.type).ljust(25),'{}'.format(ipl.shape).ljust(25),' {}'.format(ipl.activation.__name__).ljust(17),ipl.param)
		self.total_param=0
		self.non_train_param=0
		for i,obj in enumerate(self.sequence):
			print('_'*reps)
			print('{} {}({})'.format(i+1,obj.name,obj.type).ljust(25)[:25],'{}'.format(obj.shape).ljust(25),' {}'.format(obj.activation.__name__).ljust(17),obj.param)
			self.total_param+=obj.param
			if obj.__class__==layers.BatchNormalization:
				self.non_train_param+=obj.param//2
		print('='*reps)
		print("Total Params: {:,}".format(self.total_param))
		print("Trainable Params: {:,}".format(self.total_param-self.non_train_param))
		print("Non-trainable Params: {:,}".format(self.non_train_param))