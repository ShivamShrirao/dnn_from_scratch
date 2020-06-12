#!/usr/bin/env python3
from .base_layer import *
from . import seqinst


class upsampling(Layer):
    def __init__(self, input_shape=None, ksize=[2, 2], stride=[2, 2], name=None):
        # inp[batches,row,col,channels], kernels[ksz,ksz], stride[row,col]
        super().__init__()
        self.ksz = ksize[0]
        self.param = 0
        self.dtype = cp.float32
        self.type = self.__class__.__name__
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        if input_shape is None:
            input_shape = seqinst.seq_instance.get_inp_shape()
        self.batches = 1
        self.row, self.col, self.channels = input_shape
        self.out_row, self.out_col = self.row * self.ksz, self.col * self.ksz
        self.shape = (None, self.out_row, self.out_col, self.channels)
        self.activation = echo

    def forward(self, inp, training=True):
        self.input_shape = inp.shape
        return inp.repeat(2, axis=2).repeat(2, axis=1)

    def backprop(self, grads, layer=1):
        # grads[self.batches,esz,esz,self.channels],inp[self.batches,row,col,self.channels],kernels[self.ksz,self.ksz],stride[row,col]
        grads = grads.reshape(self.input_shape[0], self.row, self.ksz, self.col, self.ksz, self.channels)
        return grads.sum(axis=(2, 4), keepdims=True).reshape(self.input_shape)
