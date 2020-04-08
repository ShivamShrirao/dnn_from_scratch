#!/usr/bin/env python3
from nnet import layers
from nnet.functions import *
from nnet.optimizers import *
import pickle
from gc import collect
import time

### TO-DO- In train/fit unifunc, transpose whole data of inp at once and remove from layers.

class Sequential:
	def __init__(self):
		layers.seq_instance=self
		self.sequence=[]
		self.learning_rate=0.001
		self.dtype=np.float32

	def add(self,obj):
		self.sequence.append(obj)

	def get_inp_shape(self):
		return self.sequence[-1].shape[1:]

	def forward(self,X_inp,training=True):
		for obj in self.sequence:
			X_inp=obj.forward(X_inp,training=training)
		return X_inp

	def backprop(self,err,i):
		for obj in self.sequence[::-1]:
			err=obj.backprop(err,layer=i)
			i-=1
		return err

	def predict(self,X_inp):
		self.svd_inp=X_inp[:1].astype(self.dtype)
		return self.forward(X_inp.astype(self.dtype),training=False)

	def train_on_batch(self,X_inp,labels):
		X_inp=self.forward(X_inp.astype(self.dtype))
		err=self.del_loss(X_inp,labels.astype(self.dtype))
		self.backprop(err,self.lenseq_m1)
		self.optimizer(self.sequence,self.learning_rate,self.beta)
		return X_inp

	def not_train_on_batch(self,X_inp,labels):
		X_inp=self.forward(X_inp.astype(self.dtype))
		err=self.del_loss(X_inp,labels.astype(self.dtype))
		err=self.backprop(err,self.lenseq_m1+1)
		return X_inp,err

	def fit(self,X_inp=None,labels=None,iterator=None,batch_size=1,epochs=1,validation_data=None,shuffle=True,accuracy_metric=True,beta=0.2):
		lnxinp=len(X_inp)
		acc=0
		loss=0
		sam_time=0
		for epch in range(epochs):
			print("EPOCH:",epch+1,"/",epochs)
			if iterator==None:
				s=np.random.permutation(lnxinp)
				X_inp=X_inp[s]
				labels=labels[s]
			start=time.time()
			idx=0
			while idx<lnxinp:
				smtst=time.time()
				if iterator!=None:
					inp,y_inp=iterator.next()
				else:
					inp=X_inp[idx:idx+batch_size]
					y_inp=labels[idx:idx+batch_size]
				idx+=batch_size
				logits=self.train_on_batch(inp,y_inp)
				if accuracy_metric:
					if self.loss==cross_entropy_with_logits:
						ans=logits.argmax(axis=1)
						cor=y_inp.argmax(axis=1)
					else:
						ans=logits
						cor=y_inp
					nacc=(ans==cor).mean()
					acc =beta*nacc + (1-beta)*acc
				sample_loss=self.loss(logits=logits,labels=y_inp).mean()/10
				loss =beta*sample_loss + (1-beta)*loss
				samtm=time.time()-smtst
				sam_time=beta*samtm + (1-beta)*sam_time
				rem_sam=(lnxinp-idx)/batch_size
				eta=int(rem_sam*sam_time)
				print("\rProgress: {} / {}  - {}s - {:.2}s/sample - loss: {:.4f} - accuracy: {:.4f}".format(str(idx).rjust(6),lnxinp,eta,sam_time,sample_loss,acc),end="      _")
			end=time.time()
			print("\nEpoch time: {:.3f}s".format(end-start))
			if accuracy_metric:
				self.validate(validation_data,batch_size,beta)

	def validate(self,validation_data,batch_size,beta=0.2):
		if validation_data != None:
			VX,VY=validation_data
			lnvx=len(VX)
		else:
			lnvx=-1
		vidx=0
		vacc=0
		vloss=0
		print("Calculating Validation Accuracy....",end="")
		while vidx<=lnvx:
			inp=VX[vidx:vidx+batch_size]
			y_inp=VY[vidx:vidx+batch_size]
			vidx+=batch_size
			logits=self.predict(inp)
			if self.loss==cross_entropy_with_logits:
				ans=logits.argmax(axis=1)
				cor=y_inp.argmax(axis=1)
			else:
				ans=logits
				cor=y_inp
			vacc+=(ans==cor).sum()
			sample_loss=self.loss(logits=logits,labels=y_inp).mean()/10
			vloss=beta*sample_loss + (1-beta)*vloss
		print("\rValidation Accuracy:",str(vacc/lnvx)[:5],"- val_loss:",str(vloss)[:6])

	def free(self):			#just to free memory of large batch after predict
		X_inp=self.svd_inp
		err=self.forward(X_inp,False)
		self.backprop(err,self.lenseq_m1)
		layers.COLT.free()					# MAKE ONE TO FREE UNUSED objs IN COLT
		collect()

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
		print("[!] Load and save is bugged. You can use it for saving,loading and prediction but training further from loaded weights isn't working.")
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
		print("[!] Load and save is bugged. You can use it for saving,loading and prediction but training further from loaded weights isn't working.")
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