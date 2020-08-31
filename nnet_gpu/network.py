#!/usr/bin/env python3
from . import layers
from .functions import *
from .optimizers import *
from .stream_handler import stream_maps
from .layers import Layer

import pickle
from gc import collect
import time


# TODO- In train/fit unifunc, transpose whole data of inp at once and remove from layers.
# TODO - Divide the file up maybe. More readable.

class Sequential(Layer):
	def __init__(self):
		super().__init__(None)
		self.sequence = []
		self.learning_rate = 0.001

	def add(self, obj):
		if len(self.sequence) > 0:
			obj(self.sequence[-1])
		self.sequence.append(obj)

	def compile(self, optimizer=adam, beta=0.9, loss=cross_entropy, learning_rate=0.001):
		self.optimizer = optimizer
		self.beta = beta
		self.learning_rate = learning_rate
		self.loss = loss
		if self.loss == cross_entropy:
			self.sequence[-1].not_softmax_cross_entrp = False
			self.del_loss = del_cross_soft
		elif self.loss == mean_squared_error:
			self.del_loss = del_mean_squared_error
		self.lenseq_m1 = len(self.sequence) - 1

	def forward(self, X_inp, training=True):
		obj = self.sequence[0]
		while True:
			X_inp = obj.forward(X_inp, training=training)
			if obj.output_layers:
				obj = obj.output_layers[0]
			else:
				return X_inp

	def backprop(self, grads, do_d_inp=False):
		obj = self.sequence[-1]
		while True:
			if do_d_inp:
				do_flag = True
			else:
				do_flag = obj.input_layer is not None
			grads = obj.backprop(grads, do_d_inp=do_flag)
			if obj.input_layer:
				obj = obj.input_layer
			else:
				return grads

	def predict(self, X_inp):
		return self.forward(X_inp.astype(self.dtype, copy=False), training=False)

	def train_on_batch(self, X_inp, labels):
		X_inp = self.forward(X_inp.astype(self.dtype, copy=False))
		grads = self.del_loss(X_inp, labels.astype(self.dtype, copy=False))
		self.backprop(grads, do_d_inp=False)  # The gradients with input layer will NOT be calculated.
		self.optimizer(self.sequence, self.learning_rate, self.beta)
		return X_inp

	def not_train_on_batch(self, X_inp, labels):
		X_inp = self.forward(X_inp.astype(self.dtype, copy=False))
		grads = self.del_loss(X_inp, labels.astype(self.dtype, copy=False))
		grads = self.backprop(grads, do_d_inp=True)  # Gradients with input layer will be calculated.
		return X_inp, grads

	def fit(self, X_inp=None, labels=None, iterator=None, batch_size=1, epochs=1, validation_data=None, shuffle=True, accuracy_metric=True,
			info_beta=0.2):
		lnxinp = len(X_inp)
		acc = 0
		loss = sample_loss = 0
		sam_time = 0
		for epch in range(epochs):
			print("EPOCH:", epch + 1, "/", epochs)
			if iterator is None:
				if shuffle:
					s = cp.random.permutation(lnxinp).astype(cp.int32, copy=False)
					X_inp = X_inp[s]
					labels = labels[s]
					del s
			start = time.time()
			idx = 0
			eval_stream = stream_maps.get_next_stream()
			while idx < lnxinp:
				smtst = time.time()
				if iterator is not None:
					inp, y_inp = iterator.next()
					inp = cp.asarray(inp)
					y_inp = cp.asarray(y_inp)
				else:
					inp = cp.asarray(X_inp[idx:idx + batch_size])
					y_inp = cp.asarray(labels[idx:idx + batch_size])
				idx += inp.shape[0]
				outputs = self.train_on_batch(inp, y_inp)
				self.logit_event = cp.cuda.get_current_stream().record()
				with eval_stream:
					eval_stream.wait_event(self.logit_event)
					if accuracy_metric:
						if self.loss == cross_entropy or self.loss == mean_squared_error:
							ans = outputs.argmax(axis=1)
							cor = y_inp.argmax(axis=1)
						else:
							ans = outputs
							cor = y_inp
						nacc = (ans == cor).mean().get(eval_stream)
						acc = info_beta * nacc + (1 - info_beta) * acc
					sample_loss = self.loss(outputs=outputs, labels=y_inp).mean().get(eval_stream) / 10
					loss = info_beta * sample_loss + (1 - info_beta) * loss
					samtm = time.time() - smtst
					sam_time = info_beta * samtm + (1 - info_beta) * sam_time
					rem_sam = (lnxinp - idx) / batch_size
					eta = int(rem_sam * sam_time)
					print(
							f"\rProgress: {str(idx):>6} / {lnxinp}  - {eta}s - {sam_time:.3f}s/sample - loss: {sample_loss:.4f} - accuracy: {acc:.4f}",
							end=" -  _")
			end = time.time()
			print(f"\b\bTime: {end - start:.3f}s")
			if accuracy_metric:
				self.validate(validation_data, batch_size, info_beta)

	def validate(self, validation_data, batch_size, info_beta=0.2):
		if validation_data is not None:
			VX, VY = validation_data
			lnvx = len(VX)
		else:
			lnvx = -1
		vidx = 0
		vacc = 0
		vloss = 0
		print("Calculating Validation Accuracy....", end="")
		start = time.time()
		while vidx < lnvx:
			inp = cp.asarray(VX[vidx:vidx + batch_size])
			y_inp = cp.asarray(VY[vidx:vidx + batch_size])
			vidx += inp.shape[0]
			outputs = self.predict(inp)
			if self.loss == cross_entropy or self.loss == mean_squared_error:
				ans = outputs.argmax(axis=1)
				cor = y_inp.argmax(axis=1)
			else:
				ans = outputs
				cor = y_inp
			vacc += (ans == cor).sum()
			sample_loss = self.loss(outputs=outputs, labels=y_inp).mean() / 10
			vloss = info_beta * sample_loss + (1 - info_beta) * vloss
		end = time.time()
		print(f"\rValidation Accuracy: {(vacc / lnvx).get():.4f} - val_loss: {vloss.get():.4f} - Time: {end - start:.3f}s")

	@property
	def weights(self):
		sv_me = []
		for obj in self.sequence:
			if obj.param > 0:
				if isinstance(obj, layers.BatchNormalization):
					sv_me.append((obj.weights, obj.biases, obj.moving_mean, obj.moving_var))
				else:
					sv_me.append((obj.weights, obj.biases))  # ,obj.w_m,obj.w_v,obj.b_m,obj.b_v))
		return sv_me

	@weights.setter
	def weights(self, sv_me):
		idx = 0
		for obj in self.sequence:
			if obj.param > 0:
				if isinstance(obj, layers.BatchNormalization):
					obj.kernels, obj.biases, obj.moving_mean, obj.moving_var = sv_me[idx]
				else:
					obj.kernels, obj.biases = sv_me[idx]
					if isinstance(obj, layers.Conv2D):  # TODO - Verify isinstance works.
						obj.init_back()
				obj.weights = obj.kernels
				idx += 1

	def save_weights(self, path):  # TODO - make a proper saving mechanism.
		sv_me = self.weights
		if isinstance(path, str):
			with open(path, 'wb') as f:
				pickle.dump(sv_me, f)
		else:
			pickle.dump(sv_me, path)

	def load_weights(self, path):
		if isinstance(path, str):
			with open(path, 'rb') as f:
				sv_me = pickle.load(f)
		else:
			sv_me = pickle.load(path)
		self.weights = sv_me

	def summary(self):  # TODO - Show connections. Change print format to f""
		ipl = layers.InputLayer(self.sequence[0].input_shape)
		reps = 90
		print(chr(9149) * reps)
		print("Layer (type)".ljust(25), " Output Shape".ljust(25), "Activation".ljust(17), "Param #")
		print('=' * reps)
		print('- {}({})'.format(ipl.name, ipl.type).ljust(25), '{}'.format(ipl.shape).ljust(25),
				' {}'.format(ipl.activation.__name__).ljust(17), ipl.param)
		self.total_param = 0
		self.non_train_param = 0
		for i, obj in enumerate(self.sequence):
			print('_' * reps)
			print('{} {}({})'.format(i, obj.name, obj.type).ljust(25)[:25], '{}'.format(obj.shape).ljust(25),
					' {}'.format(obj.activation.__name__).ljust(17), obj.param)
			self.total_param += obj.param
			if obj.__class__ == layers.BatchNormalization:
				self.non_train_param += obj.param // 2
		print('=' * reps)
		print("Total Params: {:,}".format(self.total_param))
		print("Trainable Params: {:,}".format(self.total_param - self.non_train_param))
		print("Non-trainable Params: {:,}".format(self.non_train_param))
