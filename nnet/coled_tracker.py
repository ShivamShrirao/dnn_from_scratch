#!/usr/bin/env python3
import numpy as np
from gc import collect

# For a single shared large memory block to reuse and not repeat allocation

class coled_tracker:
	def __init__(self):
		self.dtype=np.float32
		self.objs=set()
		self.COLED=None

	def alloc(self,coled_size):
		if self.COLED is None:
			self.COLED=np.empty(coled_size,dtype=self.dtype)
			return self.COLED
		else:
			if self.COLED.size>=coled_size:
				return self.COLED.ravel()[:coled_size]
			else:
				self.COLED=np.empty(coled_size,dtype=self.dtype)
				self.coled=self.COLED.ravel()[:self.COLED.size].reshape(self.COLED.shape)