import numpy as np
import cupy as cp
import contextlib
from dataclasses import dataclass


@dataclass()
class CupyAvailable():
	available: bool = False
	checked: bool = False

	def __call__(self):
		#return False
		if self.checked:
			return self.available

		self.available = False
		with contextlib.suppress(cp.cuda.runtime.CUDARuntimeError):
			a = cp.array([0])
			self.available = True
		self.checked = True

		return self.available


cupy_available = CupyAvailable()


def next_larger_dtype(dtype):
	lookup = {
	    np.dtype(np.uint8): np.uint16,
	    np.dtype(np.uint16): np.uint32,
	    np.dtype(np.uint32): np.uint64,
	    np.dtype(np.uint64): None,
	    np.dtype(np.int8): np.int16,
	    np.dtype(np.int16): np.int32,
	    np.dtype(np.int32): np.int64,
	    np.dtype(np.int64): None,
	    np.dtype(np.float16): np.float64,
	    np.dtype(np.float64): np.float64,
	    np.dtype(np.float32): np.float64,
	}

	if dtype not in lookup:
		breakpoint()
		print(f"couldn't find dtype: {dtype}")
		assert (dtype in lookup)

	if not lookup[dtype]:
		print(f'converting {dtype} to {np.float64} as no larger dtype exists')
		return np.float64

	return lookup[dtype]


def dtype_from_bit(n_bits):
	assert (n_bits != 0)

	if 0 < n_bits <= 8:
		return np.uint8

	if 8 < n_bits <= 16:
		return np.uint16

	if 16 < n_bits <= 32:
		return np.uint32

	if 32 < n_bits <= 64:
		return np.uint64

	assert (False)


def generate_fast_lookup(lookup):

	kind = {
	    np.dtype(np.uint8): cp.cuda.runtime.cudaChannelFormatKindUnsigned,
	    np.dtype(np.uint16): cp.cuda.runtime.cudaChannelFormatKindUnsigned,
	    np.dtype(np.uint32): cp.cuda.runtime.cudaChannelFormatKindUnsigned,
	    np.dtype(np.uint64): cp.cuda.runtime.cudaChannelFormatKindUnsigned,
	    np.dtype(np.int8): cp.cuda.runtime.cudaChannelFormatKindSigned,
	    np.dtype(np.int16): cp.cuda.runtime.cudaChannelFormatKindSigned,
	    np.dtype(np.int32): cp.cuda.runtime.cudaChannelFormatKindSigned,
	    np.dtype(np.int64): cp.cuda.runtime.cudaChannelFormatKindSigned,
	    np.dtype(np.float16): cp.cuda.runtime.cudaChannelFormatKindFloat,
	    np.dtype(np.float32): cp.cuda.runtime.cudaChannelFormatKindFloat,
	    np.dtype(np.float64): cp.cuda.runtime.cudaChannelFormatKindFloat,
	}

	assert (lookup.ndim <= 3)
	assert (lookup.dtype in kind)

	cfd = cp.cuda.texture.ChannelFormatDescriptor(lookup.dtype.itemsize * 8, 0, 0, 0, kind[lookup.dtype])

	cuda_array = cp.cuda.texture.CUDAarray(cfd, *lookup.shape)
	rd = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray, cuArr=cuda_array)
	td = cp.cuda.texture.TextureDescriptor((cp.cuda.runtime.cudaAddressModeClamp, cp.cuda.runtime.cudaAddressModeClamp),
	                                       cp.cuda.runtime.cudaFilterModePoint, cp.cuda.runtime.cudaReadModeElementType)

	texture_object = cp.cuda.texture.TextureObject(rd, td)

	cuda_array.copy_from(lookup.T)

	return texture_object


def lookup(data_type, table_name, *index, lookup_type='CArray'):

	if lookup_type == 'CArray':
		return f'{table_name}[{{ {" , ".join(map(lambda x: f"(int){x}", index))} }}]'

	elif lookup_type == 'texture':
		n = len(index)
		return f'tex{n}D<{data_type}>({table_name}, {" , ".join(map(str, index))})'

	elif lookup_type == 'index':
		return f'{table_name}[{"][".join(map(lambda x: f"(int){x}", index))}]'

	else:
		raise ValueError(f'unknown lookup_type: {lookup_type}')
