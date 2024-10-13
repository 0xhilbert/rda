import numpy as np
import cupy as cp

import rda.utils.utils
import rda.optimized.utils
import rda.optimized.functions
import traceback

from dataclasses import dataclass, field
from typing import List, Union, Any


class Model:

	def __init__(self, components, name=None, description=None):
		self.components = components
		self.name = name
		self.description = description

	def __str__(self):
		return self.name or ' + '.join(map(str, self.components))

	def __repr__(self):
		components = map(str, self.components)

		return ' + '.join(f'{x}' for x in components)

	def __and__(self, other):
		if isinstance(other, Model):
			return Model(self.components + other.components, name=self.name + "&" + other.name)
		else:
			return Model(self.components + other, name=self.name + "&" + str(other))

	def set_coefficients(self, coefficients):
		for coef, comp in zip(coefficients, self.components):
			comp.coefficient = coef

	def describe(self):
		print(f'model: {self!s}')
		print(f'sv: {self.secret_words()}')
		print(f'pv: {self.public_words()}')
		print(f'Comps:\n{self!r}')
		print(f'Desc:\n{self.description or ""}')

	def eval(self, data, shape=None):
		R = self.components[0].eval(data)
		for c in self.components[1:]:
			R = np.add(R, c.eval(data), casting='safe')
		return R

	def eval_generator(self, data):
		# compute each element of the hypothesis and yield it
		for c in self.components:
			yield c.eval(data)

	def __len__(self):
		return len(self.components)

	def __recurse_components(self, method, *args, **kwargs):
		return [getattr(x, method)(*args, **kwargs) for x in self.components]

	def secret_words(self):
		word_instances = sum(self.__recurse_components('secret_words'), [])
		return rda.utils.utils.unique_ordered(map(str, word_instances))

	def public_words(self):
		word_instances = sum(self.__recurse_components('public_words'), [])
		return rda.utils.utils.unique_ordered(map(str, word_instances))


class ModelComponent(object):

	def __init__(self, *children):
		self.children = children
		self.coefficient = None

	def eval(self, data):
		if self.coefficient is None:
			return self.eval_impl(data)
		else:
			return self.eval_impl(data) * self.coefficient

	def __getattr__(self, name):
		if name == 'x' and len(self.children) >= 1:
			return self.children[0]

		if name == 'y' and len(self.children) >= 2:
			return self.children[1]

		if name == 'z' and len(self.children) >= 3:
			return self.children[2]

		return object.__getattribute__(self, name)

	def __add__(self, other):

		if isinstance(other, ModelComponent):
			return [self, other]

		if isinstance(other, list):
			return [self] + other

		return NotImplemented

	def __radd__(self, other):
		return self + other

	def __mul__(self, other):

		if isinstance(other, float) or isinstance(other, int):
			self.coefficient = other
			return self

		return NotImplemented

	def __rmul__(self, other):
		return self * other

	def get_kernel(self):
		raise NotImplementedError

	def __recurse_children(self, method, *args, **kwargs):
		return [getattr(x, method)(*args, **kwargs) for x in self.children]

	def __optimizable(self, words_to_check):
		words = self.secret_words() + self.public_words()
		return set(map(str, words)) == set(map(str, words_to_check))

	def __precompute(self, type, values):
		#print(f'precompute: {self!s}')
		values[str(self)] = self.eval(values)
		return type(str(self))

	def __str__(self):
		if self.coefficient is None:
			return self.to_string()
		else:
			return f'{self.to_string()}*{self.coefficient}'

	def secret_words(self):
		return sum(self.__recurse_children('secret_words'), [])

	def public_words(self):
		return sum(self.__recurse_children('public_words'), [])


# ███████  ██████  ██    ██ ██████   ██████ ███████ ███████
# ██      ██    ██ ██    ██ ██   ██ ██      ██      ██
# ███████ ██    ██ ██    ██ ██████  ██      █████   ███████
#      ██ ██    ██ ██    ██ ██   ██ ██      ██           ██
# ███████  ██████   ██████  ██   ██  ██████ ███████ ███████


@dataclass(init=False)
class Constant(ModelComponent):
	'''Models a data word'''

	def eval_impl(self, data):
		return self.x

	def get_kernel(self, tmp_variables):
		return str(self.x)

	# overload
	def secret_words(self):
		return []

	# overload
	def public_words(self):
		return []

	def to_string(self):
		return f'C({self.x})'


@dataclass(init=False)
class Word(ModelComponent):
	'''Models a data word'''

	def eval_impl(self, data):
		x = data[self.x]
		xp = cp.get_array_module(x)
		return xp.array(x)

	def get_kernel(self, tmp_variables):
		return str(self.x)

	# overload
	def secret_words(self):
		return []

	# overload
	def public_words(self):
		return []

	def __str__(self):
		return self.to_string()

	def to_string(self):
		return f'{self.x}'


@dataclass(init=False)
class Secret(Word):

	def secret_words(self):
		return [self]


@dataclass(init=False)
class Public(Word):

	def public_words(self):
		return [self]


# ███████ ██    ██ ███    ██  ██████ ████████ ██  ██████  ███    ██ ███████
# ██      ██    ██ ████   ██ ██         ██    ██ ██    ██ ████   ██ ██
# █████   ██    ██ ██ ██  ██ ██         ██    ██ ██    ██ ██ ██  ██ ███████
# ██      ██    ██ ██  ██ ██ ██         ██    ██ ██    ██ ██  ██ ██      ██
# ██       ██████  ██   ████  ██████    ██    ██  ██████  ██   ████ ███████


@dataclass(init=False)
class SBOX(ModelComponent):
	'''Models a data word through the SBOX'''

	def eval_impl(self, data):
		return rda.optimized.functions.sbox(self.x.eval(data))

	def get_kernel(self, tmp_variables):
		return tmp_variables[f'sbox[{self.x.get_kernel(tmp_variables)}]']

	def to_string(self):
		return f'S[{self.x}]'


@dataclass(init=False)
class SBOX_INV(ModelComponent):
	'''Models a data word through the inverse SBOX'''

	def eval_impl(self, data):
		return rda.optimized.functions.sbox_inv(self.x.eval(data))

	def get_kernel(self, tmp_variables):
		return tmp_variables[f'sbox_inv[{self.x.get_kernel(tmp_variables)}]']

	def to_string(self):
		return f'I[{self.x}]'


@dataclass(init=False)
class Mul(ModelComponent):
	'''Model the hamming weight of a single word'''

	def eval_impl(self, data):
		return self.x.eval(data) * self.y.eval(data)

	def get_kernel(self, tmp_variables):
		raise RuntimeError('get_kernel should never be called on reductions')

	def to_string(self):
		return f'mul({self.x},{self.y})'


@dataclass(init=False)
class XOR(ModelComponent):
	'''Model the hamming distance between two words'''

	def eval_impl(self, data):
		x_values = self.x.eval(data)
		y_values = self.y.eval(data)
		return x_values ^ y_values

	def get_kernel(self, tmp_variables):
		return tmp_variables[f'( {self.x.get_kernel(tmp_variables)} ^ {self.y.get_kernel(tmp_variables)} )']

	def to_string(self):
		return f'{self.x}^{self.y}'  # f'xor({self.x},{self.y})'


@dataclass(init=False)
class Add(ModelComponent):
	'''Model the hamming weight of a single word'''

	def eval_impl(self, data):
		return self.x.eval(data) + self.y.eval(data)

	def get_kernel(self, tmp_variables):
		return tmp_variables[f'({self.x.get_kernel(tmp_variables)})+({self.y.get_kernel(tmp_variables)})']

	def to_string(self):
		return f'add({self.x},{self.y})'


# ██████  ███████ ██████  ██    ██  ██████ ████████ ██  ██████  ███    ██ ███████
# ██   ██ ██      ██   ██ ██    ██ ██         ██    ██ ██    ██ ████   ██ ██
# ██████  █████   ██   ██ ██    ██ ██         ██    ██ ██    ██ ██ ██  ██ ███████
# ██   ██ ██      ██   ██ ██    ██ ██         ██    ██ ██    ██ ██  ██ ██      ██
# ██   ██ ███████ ██████   ██████   ██████    ██    ██  ██████  ██   ████ ███████


@dataclass(init=False)
class HW(ModelComponent):
	'''Model the hamming weight of a single word'''

	def eval_impl(self, data):
		return rda.optimized.functions.hw(self.x.eval(data))

	def get_kernel(self, tmp_variables):
		raise RuntimeError('get_kernel should never be called on HW')

	def to_string(self):
		return f'hw({self.x})'


@dataclass(init=False)
class HD(ModelComponent):
	'''Model the hamming distance between two words'''

	def eval_impl(self, data):
		x_values = self.x.eval(data)
		y_values = self.y.eval(data)
		return rda.optimized.functions.hd(x_values, y_values)

	def get_kernel(self, tmp_variables):
		raise RuntimeError('get_kernel should never be called on HD')

	def to_string(self):
		return f'hd({self.x},{self.y})'


@dataclass(init=False)
class HWSBOX(ModelComponent):
	'''Model the hamming distance between two words'''

	def eval_impl(self, data):
		x_values = self.x.eval(data)
		y_values = self.y.eval(data)
		return rda.optimized.functions.hw(rda.optimized.functions.sbox(x_values ^ y_values))

	def get_kernel(self, tmp_variables):
		raise RuntimeError('get_kernel should never be called on HWSBOX')

	def to_string(self):
		return f'hw(sbox({self.x}^{self.y}))'


# ██████   █████  ██████  ███████ ███████ ██████
# ██   ██ ██   ██ ██   ██ ██      ██      ██   ██
# ██████  ███████ ██████  ███████ █████   ██████
# ██      ██   ██ ██   ██      ██ ██      ██   ██
# ██      ██   ██ ██   ██ ███████ ███████ ██   ██


def model_from_string(string, columns, vs, gs):

	local_vars = {
	    'xor': XOR,
	    'sbox': SBOX,
	    'sboxinv': SBOX_INV,
	    'mul': Mul,
	    'add': Add,
	    'C': Constant,
	    'hd': HD,
	    'hw': HW,
	    'hwsbox': HWSBOX,
	}

	try:
		import rda.power.project.model_definitions
		local_vars.update(rda.power.project.model_definitions.get_models(vs, gs))
	except:
		pass

	# get the named wods
	vs_str = list(map(str, vs))
	gs_str = list(map(str, vs))

	# add all words to the parser
	local_vars.update({str(v): v for v in vs})
	local_vars.update({str(g): g for g in gs})

	# also add columns of data frame to the parser
	local_vars.update({c: Public(c) for c in columns if c not in vs_str and c not in gs_str})

	# TODO: export
	while True:
		try:
			exec(f'model = {string}', local_vars)
			break
		#except NameError as e:
		#	var = str(e).split('\'')[1]  # todo fix with python 3.10
		#	print(f' converting {var} to Word ')
		#	local_vars[var] = Word(var)
		except Exception as e:
			print(f'Converting {string!r} to model failed! raised exception: {e!r}')
			print(traceback.format_exc())
			exit(-1)

	model = local_vars['model']

	if isinstance(model, list):
		model = Model(model)

	if isinstance(model, ModelComponent):
		model = Model([model])

	if not isinstance(model, Model):
		raise RuntimeError('Model is not of type model how?')

	return model
