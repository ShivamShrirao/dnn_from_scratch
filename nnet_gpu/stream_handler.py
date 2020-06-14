#!/usr/bin/env python3
import cupy as cp

N_STREAMS = 16


class stream_mapper:
	def __init__(self):
		self.default_stream = cp.cuda.Stream.null  # get_current_stream()
		self.streams = [cp.cuda.Stream(non_blocking=True) for i in range(N_STREAMS)]
		self.idx = 0

	def get_next_stream(self):
		self.idx = (self.idx + 1) % len(self.streams)
		return self.streams[self.idx]
	# return self.default_stream


stream_maps = stream_mapper()
