import scipy
import cupy as cp
import numpy as np
import rda.optimized.utils


def confidence_interval(r, n, level):
	xp = cp.get_array_module(r)

	# Fisher’s r-to-z transformation
	z = xp.arctanh(r)

	assert (n >= 3)

	# expected z value
	zse = xp.sqrt(1.0 / (n - 3.0))

	# the sigma needed to reach the given confidence (two sided)
	z_sigma = scipy.stats.norm.ppf(level + (1 - level) / 2)

	# bounds in z
	z_upper = z + zse * z_sigma
	z_lower = z - zse * z_sigma

	# bounds transformed back to r
	r_upper = xp.tanh(z_upper)
	r_lower = xp.tanh(z_lower)

	# two sided pvalue
	# pv = scipy.stats.norm.sf(abs(z), scale=zse) * 2

	# formulas taken from scipy:
	df = n - 2
	t_squared = r**2 * (df / ((1.0 - r) * (1.0 + r)))
	pv = scipy.special.betainc(0.5 * df, 0.5, df / (df + t_squared))

	return (r_lower, r_upper, pv)


def compute_intermediates(x, axis):
	xp = cp.get_array_module(x)

	n = x.shape[axis]

	sum_x = x.sum(axis=axis)

	sum_xx = xp.square(x, dtype=rda.optimized.utils.next_larger_dtype(x.dtype)).sum(axis=axis)

	scale_x = n * sum_xx - sum_x**2

	return sum_x, sum_xx, scale_x


def rolling_sum(x, n, axis):
	y = x.cumsum(axis=axis)

	return y[..., n:] - y[..., :-n]  # todo use axis as variable


def compute_rolling_intermediates(x, n, axis):
	xp = cp.get_array_module(x)

	sum_x = rolling_sum(x, n, axis=axis)

	sum_xx = rolling_sum(xp.square(x, dtype=rda.optimized.utils.next_larger_dtype(x.dtype)), n, axis=axis)

	scale_x = n * sum_xx - sum_x**2

	return sum_x, sum_xx, scale_x


class PearsonRCalculator:

	def __init__(self, y):
		xp = cp.get_array_module(y)

		self.y = y.astype(xp.float64)
		self.n = y.shape[0]

		self.sum_y, _, self.scale_y = compute_intermediates(self.y, axis=0)

	def compute(self, X, confidence=False, level=0.95):
		xp = cp.get_array_module(X)

		sum_x, _, scale_x = compute_intermediates(X, axis=1)

		with np.errstate(divide='ignore'):
			rho = (self.n * X.dot(self.y) - sum_x * self.sum_y) / xp.sqrt(scale_x * self.scale_y)

		if xp.any(xp.abs(rho) > 1) and not xp.any(xp.isnan(rho) | xp.isinf(rho)):
			breakpoint()

		if not confidence:
			return rho

		return rho, *confidence_interval(rho, self.n, level=level)


class PearsonRCalculatorRolling:

	def __init__(self, y, window):
		xp = cp.get_array_module(y)

		self.y = y
		self.n = window

		self.sum_y, _, self.scale_y = compute_rolling_intermediates(self.y, self.n, axis=0)

	def compute(self, X, confidence=False, level=0.95):
		xp = cp.get_array_module(X)

		sum_x, _, scale_x = compute_rolling_intermediates(X, self.n, axis=1)

		sum_xy = rolling_sum(X * self.y, self.n, axis=1)

		rho = (self.n * sum_xy - (sum_x * self.sum_y.T)) / xp.sqrt(scale_x * self.scale_y)

		if not confidence:
			return rho

		return rho, *confidence_interval(rho, self.n, level=level)


class IterativePearsonRCalculator:

	def __init__(self, y):
		xp = cp.get_array_module(y)

		self.y = y
		self.n = y.shape[0]
		self.i = 0

		self.sum_y, _, self.scale_y = compute_intermediates(self.y, axis=0)

		self.reset()

	def reset(self):
		self.x_dot_y = None
		self.sum_x = None
		self.sum_xx = None

	def compute_slice(self, X):
		xp = cp.get_array_module(X)

		if self.x_dot_y is None:
			self.x_dot_y = xp.zeros((X.shape[0],))
			self.sum_x = xp.zeros((X.shape[0],))
			self.sum_xx = xp.zeros((X.shape[0],))

		assert (self.x_dot_y.shape[0] == X.shape[0])

		tmp_sum_x, tmp_sum_xx, _ = compute_intermediates(X, axis=1)

		self.sum_x += tmp_sum_x
		self.sum_xx += tmp_sum_xx
		self.x_dot_y += X.dot(self.y[self.i:self.i + X.shape[1]])

		self.i += X.shape[1]

	def finalize(self, confidence=False, level=0.95):
		xp = cp.get_array_module(self.x_dot_y)

		rho = (self.n * self.x_dot_y - self.sum_x * self.sum_y) / xp.sqrt(
		    (self.n * self.sum_xx - self.sum_x**2) * self.scale_y)

		if not confidence:
			return rho

		return rho, *confidence_interval(rho, self.n, level=level)


def test_pearsonr():
	''' make sure the implementation works correctly -> use scipy as reference'''

	x = np.sin(np.arange(100))
	y1 = np.sin(np.arange(1, 101))
	y2 = np.sin(np.arange(2, 102))

	r1 = scipy.stats.pearsonr(x, y1)
	r2 = scipy.stats.pearsonr(x, y2)
	rho, r_lower, r_upper, pv = PearsonRCalculator(x).compute(np.vstack([y1, y2]), confidence=True)

	assert (np.all(np.isclose(rho, [r1.statistic, r2.statistic])))
	assert (np.all(np.isclose(pv, [r1.pvalue, r2.pvalue])))
	assert (np.all(np.isclose(r_lower, [r1.confidence_interval(0.95).low, r2.confidence_interval(0.95).low])))
	assert (np.all(np.isclose(r_upper, [r1.confidence_interval(0.95).high, r2.confidence_interval(0.95).high])))


test_pearsonr()
