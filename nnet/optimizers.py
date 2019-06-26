#!/usr/bin/env python3
import numpy as np

### CAN TURN THESE INTO CLASSES

def iterative(sequence,learning_rate=0.01):
	for obj in sequence:
		if obj.param>0:
			obj.weights-=learning_rate*obj.d_c_w
			obj.biases-=learning_rate*obj.d_c_b

def momentum(sequence,learning_rate=0.01,beta1=0.9,weight_decay=0.0005):	# will have to specify it
	for obj in sequence:
		if obj.param>0:
			obj.w_m=beta1*obj.w_m - learning_rate*obj.d_c_w - weight_decay*learning_rate*obj.weights
			obj.weights+=obj.w_m
			obj.b_m=beta1*obj.b_m - learning_rate*obj.d_c_b - weight_decay*learning_rate*obj.biases
			obj.biases+=obj.b_m

def rmsprop(sequence,learning_rate=0.001,beta1=0.9,epsilon=1e-8):
	for obj in sequence:
		if obj.param>0:
			obj.w_v=beta1*obj.w_v + (1-beta1)*(obj.d_c_w**2)
			obj.weights-=learning_rate*(obj.d_c_w/np.sqrt(obj.w_v+epsilon))
			obj.b_v=beta1*obj.b_v + (1-beta1)*(obj.d_c_b**2)
			obj.biases-=learning_rate*(obj.d_c_b/np.sqrt(obj.b_v+epsilon))

def adagrad(sequence,learning_rate=0.01,beta1=0.9,epsilon=1e-8):
	for obj in sequence:
		if obj.param>0:
			obj.w_v+=(obj.d_c_w**2)
			obj.weights-=learning_rate*(obj.d_c_w/np.sqrt(obj.w_v+epsilon))
			obj.b_v+=(obj.d_c_b**2)
			obj.biases-=learning_rate*(obj.d_c_b/np.sqrt(obj.b_v+epsilon))

def adam(sequence,learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8,decay=0):		# decay not functional rn
	for obj in sequence:
		if obj.param>0:
			# Update weights
			obj.w_m=beta1*obj.w_m + (1-beta1)*obj.d_c_w
			obj.w_v=beta2*obj.w_v + (1-beta2)*(obj.d_c_w**2)
			mcap=obj.w_m/(1-beta1)
			vcap=obj.w_v/(1-beta2)
			obj.d_c_w=mcap/(np.sqrt(vcap)+epsilon)
			obj.weights-=learning_rate*obj.d_c_w
			# Update biases
			obj.b_m=beta1*obj.b_m + (1-beta1)*obj.d_c_b
			obj.b_v=beta2*obj.b_v + (1-beta2)*(obj.d_c_b**2)
			mcap=obj.b_m/(1-beta1)
			vcap=obj.b_v/(1-beta2)
			obj.d_c_b=mcap/(np.sqrt(vcap)+epsilon)
			obj.biases-=learning_rate*obj.d_c_b

def adamax(sequence,learning_rate=0.002,beta1=0.9,beta2=0.999,epsilon=1e-8):
	for obj in sequence:
		if obj.param>0:
			# Update weights
			obj.w_m=beta1*obj.w_m + (1-beta1)*obj.d_c_w
			obj.w_v=np.maximum(beta2*obj.w_v,abs(obj.d_c_w))
			obj.weights-=(learning_rate/(1-beta1))*(obj.w_m/(obj.w_v+epsilon))
			# Update biases
			obj.b_m=beta1*obj.b_m + (1-beta1)*obj.d_c_b
			obj.b_v=np.maximum(beta2*obj.b_v,abs(obj.d_c_b))
			obj.biases-=(learning_rate/(1-beta1))*(obj.b_m/(obj.b_v+epsilon))

def adadelta(sequence,learning_rate=0.01,beta1=0.9,epsilon=1e-8):
	for obj in sequence:
		if obj.param>0:
			obj.w_v=beta1*obj.w_v + (1-beta1)*(obj.d_c_w**2)
			obj.d_c_w=np.sqrt((obj.w_m+epsilon)/(obj.w_v+epsilon))*obj.d_c_w
			obj.w_m=beta1*obj.w_m + (1-beta1)*(obj.d_c_w**2)
			obj.weights-=obj.d_c_w

			obj.b_v=beta1*obj.b_v + (1-beta1)*(obj.d_c_b**2)
			obj.d_c_b=np.sqrt((obj.b_m+epsilon)/(obj.b_v+epsilon))*obj.d_c_b
			obj.b_m=beta1*obj.b_m + (1-beta1)*(obj.d_c_b**2)
			obj.biases-=obj.d_c_b