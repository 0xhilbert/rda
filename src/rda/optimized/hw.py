# This function was adapted from a Stack Overflow post by Mad Physicist.
# Source: https://stackoverflow.com/a/68943135
# Licensed under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
# Adapted the np.iinfo(t).max part and converted the result to uint8 finally moved it to own function in this file

import numpy as np


def hw_impl(x):

	# taken from
	# https://stackoverflow.com/questions/63954102/numpy-vectorized-way-to-count-non-zero-bits-in-array-of-integers
	# outperforms other approaches by orders of magnitude

	t = x.dtype.type
	mask = np.iinfo(t).max
	s55 = t(0x5555555555555555 & mask)
	s33 = t(0x3333333333333333 & mask)
	s0F = t(0x0F0F0F0F0F0F0F0F & mask)
	s01 = t(0x0101010101010101 & mask)

	x = x - ((x >> t(1)) & s55)

	x = (x & s33) + ((x >> t(2)) & s33)
	x = (x + (x >> t(4))) & s0F
	x = (x * s01) >> t(8 * (x.itemsize - 1))

	return x.astype(np.uint8)
