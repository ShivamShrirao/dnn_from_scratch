#!/usr/bin/env python3
import cupy as cp

# Both kernels from chainer. May make more efficient by calculating indices once and reusing them all time,
# like done in CPU version, instead of having to calculate everytime.
im2col = cp.ElementwiseKernel(
		'raw T inp, int32 row, int32 col, int32 out_row, int32 out_col,'
		'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
		'int32 dy, int32 dx',
		'T coled',
		'''
			int c0 = i / (kh * kw * out_row * out_col);		// select channel
			int ky = i / (kw * out_row * out_col) % kh;		// select kernel y
			int kx = i / (out_row * out_col) % kw;			// select kernel x
			int out_y = i / out_col % out_row;				// select output y
			int out_x = i % out_col;						// select output x
			int in_y = ky * dy + out_y * sy - ph;
			int in_x = kx * dx + out_x * sx - pw;
			if (in_y >= 0 && in_y < row && in_x >= 0 && in_x < col) {	// if in image bounds
				coled = inp[col * (in_y + row * c0) + in_x];	// choose pixel
			} else {
				coled = 0;						// pad with 0
			}
		''',
		'im2col')

col2im = cp.ElementwiseKernel(
		'raw T coled, int32 row, int32 col, int32 out_row, int32 out_col,'
		'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
		'int32 dy, int32 dx',
		'T inp',
		'''
			int c0 = i / (row * col);
			int y  = i / col % row;
			int x  = i % col;
			T val = 0;
			for (int ky = 0; ky < kh; ++ky) {
				int out_y = (y + ph - ky * dy);
				if (0 > out_y || out_y >= out_row * sy) continue;
				if (out_y % sy != 0) continue;
				out_y /= sy;
				for (int kx = 0; kx < kw; ++kx) {
					int out_x = (x + pw - kx * dx);
					if (0 > out_x || out_x >= out_col * sx) continue;
					if (out_x % sx != 0) continue;
					out_x /= sx;
					int k = out_y + out_row * (kx + kw * (ky + kh * c0));
					val = val + coled[out_x + out_col * k];
				}
			}
			inp = val;
		''',
		'col2im')


class emptyHelper:
	def __init__(self, shape):
		self.shape = shape
