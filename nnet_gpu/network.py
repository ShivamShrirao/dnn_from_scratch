#!/usr/bin/env python3
from . import layers
from .functions import *
from .optimizers import *
from .stream_handler import stream_maps

import pickle
from gc import collect
import time

### TO-DO- In train/fit unifunc, transpose whole data of inp at once and remove from layers.

class Sequential:
	def __init__(self):
		layers.seqinst.seq_instance=self
		self.sequence=[]
		self.learning_rate=0.001
		self.dtype=cp.float32

	def add(self,obj):
		if len(self.sequence)>0:
			obj(self.sequence[-1])
		self.sequence.append(obj)

	def get_inp_shape(self):
		return self.sequence[-1].shape[1:]

	def forward(self,X_inp,training=True):
		for obj in self.sequence:
			X_inp=obj.forward(X_inp,training=training)
		return X_inp

	def backprop(self,grads,i):
		for obj in self.sequence[::-1]:
			grads=obj.backprop(grads,layer=i)
			i-=1
		return grads

	def predict(self,X_inp):
		return self.forward(X_inp.astype(self.dtype,copy=False),training=False)

	def train_on_batch(self,X_inp,labels):
		X_inp=self.forward(X_inp.astype(self.dtype,copy=False))
		grads=self.del_loss(X_inp,labels.astype(self.dtype,copy=False))
		self.backprop(grads,self.lenseq_m1)
		self.optimizer(self.sequence,self.learning_rate,self.beta)
		return X_inp

	def not_train_on_batch(self,X_inp,labels):
		X_inp=self.forward(X_inp.astype(self.dtype,copy=False))
		grads=self.del_loss(X_inp,labels.astype(self.dtype,copy=False))
		grads=self.backprop(grads,self.lenseq_m1+1)
		return X_inp,grads

	def fit(self,X_inp=None,labels=None,iterator=None,batch_size=1,epochs=1,validation_data=None,shuffle=True,accuracy_metric=True,infobeta=0.2):
		lnxinp=len(X_inp)
		acc=0
		loss=sample_loss=0
		sam_time=0
		for epch in range(epochs):
			print("EPOCH:",epch+1,"/",epochs)
			if iterator==None:
				if shuffle:
					s=cp.random.permutation(lnxinp).astype(cp.int32,copy=False)
					X_inp=X_inp[s]
					labels=labels[s]
					del s
			start=time.time()
			idx=0
			eval_stream=stream_maps.get_next_stream()
			while idx<lnxinp:
				smtst=time.time()
				if iterator!=None:
					inp,y_inp=iterator.next()
					inp=cp.asarray(inp)
					y_inp=cp.asarray(y_inp)
				else:
					inp=cp.asarray(X_inp[idx:idx+batch_size])
					y_inp=cp.asarray(labels[idx:idx+batch_size])
				idx+=inp.shape[0]
				logits=self.train_on_batch(inp,y_inp)
				self.logit_event=cp.cuda.get_current_stream().record()
				with eval_stream:
					eval_stream.wait_event(self.logit_event)
					if accuracy_metric:
						if self.loss==cross_entropy_with_logits:
							ans=logits.argmax(axis=1)
							cor=y_inp.argmax(axis=1)
						else:
							ans=logits
							cor=y_inp
						nacc=(ans==cor).mean().get(eval_stream)
						acc =infobeta*nacc + (1-infobeta)*acc
					sample_loss=self.loss(logits=logits,labels=y_inp).mean().get(eval_stream)/10
					loss =infobeta*sample_loss + (1-infobeta)*loss
					samtm=time.time()-smtst
					sam_time=infobeta*samtm + (1-infobeta)*sam_time
					rem_sam=(lnxinp-idx)/batch_size
					eta=int(rem_sam*sam_time)
					print(f"\rProgress: {str(idx):>6} / {lnxinp}  - {eta}s - {sam_time:.3f}s/sample - loss: {sample_loss:.4f} - accuracy: {acc:.4f}",end=" -  _")
			end=time.time()
			print(f"\b\bTime: {end-start:.3f}s")
			if accuracy_metric:
				self.validate(validation_data,batch_size,infobeta)

	def validate(self,validation_data,batch_size,infobeta=0.2):
		if validation_data != None:
			VX,VY=validation_data
			lnvx=len(VX)
		else:
			lnvx=-1
		vidx=0
		vacc=0
		vloss=0
		print("Calculating Validation Accuracy....",end="")
		start=time.time()
		while vidx<lnvx:
			inp=cp.asarray(VX[vidx:vidx+batch_size])
			y_inp=cp.asarray(VY[vidx:vidx+batch_size])
			vidx+=inp.shape[0]
			logits=self.predict(inp)
			if self.loss==cross_entropy_with_logits:
				ans=logits.argmax(axis=1)
				cor=y_inp.argmax(axis=1)
			else:
				ans=logits
				cor=y_inp
			vacc+=(ans==cor).sum()
			sample_loss=self.loss(logits=logits,labels=y_inp).mean()/10
			vloss=infobeta*sample_loss + (1-infobeta)*vloss
		end=time.time()
		print(f"\rValidation Accuracy: {(vacc/lnvx).get():.4f} - val_loss: {vloss.get():.4f} - Time: {end-start:.3f}s")

	def compile(self,optimizer=adam,beta=0.9,loss=cross_entropy_with_logits,learning_rate=0.001):
		self.optimizer=optimizer
		self.beta=beta
		self.learning_rate=learning_rate
		self.loss=loss
		if self.loss==cross_entropy_with_logits:
			self.sequence[-1].not_softmax_cross_entrp=False
			self.del_loss=del_cross_soft
		elif self.loss==mean_squared_error:
			self.del_loss=del_mean_squared_error
		self.lenseq_m1=len(self.sequence)-1

	def save_weights(self,path):	# has problems if u wanna train the network further. Need to fix that.
		sv_me=[]					# OK for just validation and prediction.
		for obj in self.sequence:	# FIX: Prolly d_ker is seeing different kernel
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
					if obj.__class__==layers.conv2d:
						obj.d_inp.kernels=obj.weights
						obj.init_back()
				obj.kernels=obj.weights
				idx+=1

	def summary(self):
		ipl=layers.InputLayer(self.sequence[0].input_shape)
		reps=90
		print(chr(9149)*reps)
		print("Layer (type)".ljust(25)," Output Shape".ljust(25),"Activation".ljust(17),"Param #")
		print('='*reps)
		print('- {}({})'.format(ipl.name,ipl.type).ljust(25),'{}'.format(ipl.shape).ljust(25),' {}'.format(ipl.activation.__name__).ljust(17),ipl.param)
		self.total_param=0
		self.non_train_param=0
		for i,obj in enumerate(self.sequence):
			print('_'*reps)
			print('{} {}({})'.format(i,obj.name,obj.type).ljust(25)[:25],'{}'.format(obj.shape).ljust(25),' {}'.format(obj.activation.__name__).ljust(17),obj.param)
			self.total_param+=obj.param
			if obj.__class__==layers.BatchNormalization:
				self.non_train_param+=obj.param//2
		print('='*reps)
		print("Total Params: {:,}".format(self.total_param))
		print("Trainable Params: {:,}".format(self.total_param-self.non_train_param))
		print("Non-trainable Params: {:,}".format(self.non_train_param))