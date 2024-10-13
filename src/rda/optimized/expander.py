import numpy as np


class SimpleExpander:

	def __init__(self, dtype, word_count, word_stride):
		self.__dtype = dtype
		self.__word_count = word_count
		self.__word_stride = word_stride

	def __call__(self, x):
		return x.view(self.__dtype).reshape(x.shape[0], -1)[:, :self.__word_count:self.__word_stride]

	#def map(self, x):
	#	breakpoint()
	#	return x.view(self.dtype)[:, :self.word_count:self.word_stride]


class ExtendedExpander:

	def __init__(self, word_bits, word_count, word_stride):
		self.__word_count = word_count
		self.__shift = np.arange(0, 8, word_bits, dtype=np.uint8)
		self.__mask = (np.array((1 << word_bits) - 1, dtype=np.uint8).reshape((-1, 1)) << self.__shift)[0]
		self.__word_stride = word_stride

	def __call__(self, x):
		return ((x.view(np.uint8).reshape(
		    (-1, 1)) & self.__mask) >> self.__shift).reshape(x.shape[0], -1)[:, :self.__word_count:self.__word_stride]


def create_expander(word_bits, word_count, word_stride=1):
	'''create expander to transform uint8 stream to the desired word_bits width and word_count '''
	# check if the expander yields a 'nice' data type
	d = {
	    8: np.uint8,
	    16: np.uint16,
	    32: np.uint32,
	    64: np.uint64,
	}
	if word_bits in d:
		return SimpleExpander(d[word_bits], word_count, word_stride)

	if word_bits not in [1, 2, 4]:
		raise NotImplementedError('the requested word bit width is not implemented')

	return ExtendedExpander(word_bits, word_count, word_stride)


assert (np.all(create_expander(2, 8)(np.array([[0x8, 0x33]], dtype=np.uint8)) == np.array([[0, 2, 0, 0, 3, 0, 3, 0]])))
assert (np.all(create_expander(2, 6)(np.array([[0x8, 0x33]], dtype=np.uint8)) == np.array([[0, 2, 0, 0, 3, 0]])))
assert (np.all(create_expander(4, 4)(np.array([[0x8, 0x33]], dtype=np.uint8)) == np.array([[8, 0, 3, 3]])))
assert (np.all(create_expander(8, 2)(np.array([[0x8, 0x33]], dtype=np.uint8)) == np.array([[8, 0x33]])))
assert (np.all(create_expander(16, 1)(np.array([[0x8, 0x33]], dtype=np.uint8)) == np.array([[0x3308]])))
