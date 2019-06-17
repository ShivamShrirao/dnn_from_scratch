#!/usr/bin/env python3
import layers
from functions import *

class Sequential:
	def __init__(self):
		layers.seq_instance=self
		self.seq=[]
		self.forw_seq=[]
		self.back_seq=[]

	def add(self,obj):
		self.seq.append(obj)

	def get_inp_shape(self):
		return self.seq[-1].shape[1:]
