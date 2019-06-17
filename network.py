#!/usr/bin/env python3
import layers
from functions import *

class Sequential:
	def __init__(self):
		layers.seq_instance=self
		self.sequence=[]
		self.forw_seq=[]
		self.back_seq=[]

	def add(self,obj):
		self.sequence.append(obj)

	def get_inp_shape(self):
		return self.sequence[-1].shape[1:]

	def run(self,inp):
		for obj in self.sequence:
			inp=obj.forward(inp)
		return inp

	def fit(self,inp):
		inp=self.run(inp)
		return inp

	def compile(self,optimizer=None,loss=None):
		optimizer

	def summary(self):
		ipl=InputLayer(self.sequence[0].input_shape)
		reps=100
		print('_'*reps)
		print("Name (type)".ljust(32)," Output Shape".ljust(25),"Param #")
		print('='*reps)
		print('{} ({})'.format(ipl.name,ipl.type).ljust(32),'{}'.format(ipl.shape).ljust(25),ipl.param)
		self.total_param=0
		for i in self.sequence:
			print('_'*reps)
			print('{} ({})'.format(i.name,i.type).ljust(32),'{}'.format(i.shape).ljust(25),i.param)
			self.total_param+=i.param
		print('='*reps)
		print("Total Params:",self.total_param)

class InputLayer:
	def __init__(self,shape):
		self.name='input_layer'
		self.type=self.__class__.__name__
		self.shape=(None,*shape)
		self.param=0
