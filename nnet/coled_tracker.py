#!/usr/bin/env python3
import numpy as np
from gc import collect

# For a single shared large memory block to reuse and not repeat allocation

class coled_tracker:
	def __init__(self):
		self.dtype=np.float32
		self.objs=set()
		self.COLED=None

	def alloc(self,coled_size,obj):
		if self.COLED is None:
			self.COLED=np.empty(coled_size,dtype=self.dtype)
			for oo in self.objs:
				oo.coled=self.COLED.ravel()[:oo.coled.size].reshape(oo.coled.shape)
			self.objs.add(obj)
			return self.COLED
		else:
			if self.COLED.size>=coled_size:
				self.objs.add(obj)
				return self.COLED.ravel()[:coled_size]
			else:
				self.COLED=np.empty(coled_size,dtype=self.dtype)
				for oo in self.objs:
					oo.coled=self.COLED.ravel()[:oo.coled.size].reshape(oo.coled.shape)
				self.objs.add(obj)
				return self.COLED.ravel()[:coled_size]

	def free(self):
		obs=list(self.objs)
		mx=obs[0]
		for oo in obs:
			if oo.coled.nbytes>mx.coled.nbytes:
				mx=oo
		if self.COLED.nbytes>mx.coled.nbytes:
			self.COLED=np.empty(mx.coled.size,dtype=self.dtype)
			for oo in self.objs:
				oo.coled=self.COLED.ravel()[:oo.coled.size].reshape(oo.coled.shape)
		collect()