#!/usr/bin/env python3
import layers
from functions import *

class Sequential:
	def __init__(self):
		layers.seq_instance=self
		self.sequence=[]
		self.learning_rate=0.001

	def add(self,obj):
		self.sequence.append(obj)

	def get_inp_shape(self):
		return self.sequence[-1].shape[1:]

	def predict(self,X_inp):
		for obj in self.sequence:
			X_inp=obj.forward(X_inp)
		return X_inp

	def fit(self,X_inp,labels):
		for obj in self.sequence:
			X_inp=obj.forward(X_inp)
		err=self.del_loss(X_inp,labels)
		i=self.seq_len_m1
		for obj in self.sequence[::-1]:
			err=obj.backprop(err,layer=i)
			i-=1
		self.optimizer(self.sequence,self.learning_rate)
		return X_inp

	def compile(self,optimizer=iterative,loss=None,learning_rate=0.001):
		self.optimizer=optimizer
		self.learning_rate=learning_rate
		self.loss=loss
		if self.loss==cross_entropy_with_logits:
			self.sequence[-1].cross=True
			self.del_loss=del_cross_soft
		elif self.loss==mean_squared_error:
			self.del_loss=del_mean_squared_error
		self.seq_len_m1=len(self.sequence)-1

	def summary(self):
		ipl=layers.InputLayer(self.sequence[0].input_shape)
		reps=90
		print(chr(9149)*reps)
		print("Name (type)".ljust(30)," Output Shape".ljust(25),"Activation".ljust(17),"Param #")
		print('='*reps)
		print('{} ({})'.format(ipl.name,ipl.type).ljust(30),'{}'.format(ipl.shape).ljust(25),' {}'.format(ipl.activation.__name__).ljust(17),ipl.param)
		self.total_param=0
		for obj in self.sequence:
			print('_'*reps)
			print('{} ({})'.format(obj.name,obj.type).ljust(30),'{}'.format(obj.shape).ljust(25),' {}'.format(obj.activation.__name__).ljust(17),obj.param)
			self.total_param+=obj.param
		print('='*reps)
		print("Total Params:",self.total_param)