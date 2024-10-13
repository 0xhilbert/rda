import numpy as np
import cupy as cp
import cupyx as cpx
import pandas as pd

import scipy.stats
import rda.optimized.pearsonr
import contextlib

import time
from dataclasses import dataclass
from tqdm import tqdm

from cupyx.profiler import benchmark

import time
import rda.power
import itertools
import rda.optimized.utils

import rda.power.model

from enum import Enum


class CPA_Normalization(Enum):
	NONE = 1
	INLINE = 2
	POST = 3


#  ██████ ██████  ██    ██      ██████  ██████  ██████  ███████
# ██      ██   ██ ██    ██     ██      ██    ██ ██   ██ ██
# ██      ██████  ██    ██     ██      ██    ██ ██   ██ █████
# ██      ██      ██    ██     ██      ██    ██ ██   ██ ██
#  ██████ ██       ██████       ██████  ██████  ██████  ███████
#
# This can handle all models as long as the search space is not large
#


def fill_secret_values(ctx_power, secret_words, xp):

	bits = ctx_power.secret_word_ctx.bits

	base = xp.arange(2**bits, dtype=ctx_power.secret_word_ctx.dtype)

	combinations = xp.meshgrid(*[base] * len(secret_words), indexing="ij", copy=False)

	n_candidates = 2**(bits * len(secret_words))

	return {w: combinations[i].reshape(-1,) for i, w in enumerate(secret_words)}, n_candidates


def fill_public_values(df, ctx_power, public_words, xp):
	if xp == cp:
		return {w: xp.asarray(df[w].values) for w in public_words}
	else:
		return {w: df[w].values for w in public_words}


def compute_cpa(df, ctx_power, measurements_column, model, n_step):
	# c1 = compute_split_rho(df, ctx_power, measurements_column, model, n_step)

	n_step = 1000

	# select numpy as default
	xp = np

	# the measurements with which we perform the correlation
	measurements = scipy.stats.zscore(df[measurements_column].values)

	# if we have a gpu available move the data to the gpu
	if False and rda.optimized.utils.cupy_available():
		measurements = cp.asarray(measurements).astype(np.float64)
		xp = cp

	# the secret value candidates are iterated throughout the value space
	secret_values, n_candidates = fill_secret_values(ctx_power, model.secret_words(), xp)

	# the public values are served from the data set
	public_values = fill_public_values(df, ctx_power, model.public_words(), xp)

	# generate the optimized pearson r calculator
	pearsonr = rda.optimized.pearsonr.PearsonRCalculator(measurements)

	# custom data adapter so we can change the axis of the data
	# this allows the broadcasts within the model to do the correct
	# thing when known and unknown values collide
	# futhermore we slice the input candidates to not exceed memory
	@dataclass(frozen=True)
	class DataAdapter:
		# start of the slice
		n_start: int
		n_step: int

		def __getitem__(self, key):
			# check if a unknown value
			if key in secret_values:
				return secret_values[key][self.n_start:self.n_start + self.n_step, np.newaxis]

			# or a known value
			elif key in public_values:
				return public_values[key][np.newaxis, :]

			# huh? error case
			else:
				print(f"got key not in either the known nor unknown values: {key} is the model implemented correctly?")
				assert False

	# resulting correlation coefficients
	corrcoefs = xp.zeros((n_candidates,), dtype=np.float64)

	# iterate over the sliced unknown values and compute their corrcoefs
	for n_start in tqdm(range(0, corrcoefs.shape[0], n_step), desc="CPA", leave=False):

		# compute the hypothesis
		hypothesis_matrix = model.eval(DataAdapter(n_start, n_step))

		# correlate the hypothesis with the observations
		corrcoefs[n_start:n_start + n_step] = pearsonr.compute(hypothesis_matrix)

	# remove NaNs
	corrcoefs = xp.nan_to_num(corrcoefs, copy=False, nan=-xp.inf, posinf=-xp.inf, neginf=-xp.inf)

	# get the sorted index of the the candidates
	index = xp.argsort(corrcoefs)[::-1]

	# sort the corrcoefs
	corrcoefs = corrcoefs[index]

	# sort the candidates
	secret_values = {w: secret_values[w][index] for w in model.secret_words()}

	# if we worked on the gpu copy the data back
	if xp == cp:
		corrcoefs = cp.asnumpy(corrcoefs)
		secret_values = {k: cp.asnumpy(v) for k, v in secret_values.items()}

	# generate a result data frame
	results = pd.DataFrame(secret_values)

	# append the corrcoefs
	results["rho"] = corrcoefs

	# get the ground truth
	ground_truth = df[model.secret_words()].iloc[0].values

	# get the rank if contained
	rank = np.where(np.all(results[model.secret_words()].values == ground_truth, axis=1))[0]

	# we couldn't find the rank -> lets estimate
	if len(rank) == 0:
		results['rank'] = -1
		results['rho_gt'] = np.NaN
		results['est'] = True
	else:
		results['rank'] = rank[0]
		results['rho_gt'] = results['rho'].iloc[rank[0]]
		results['est'] = False

	return results


#  ██████  ██████  ██    ██      ██████  ██████  ██████  ███████
# ██       ██   ██ ██    ██     ██      ██    ██ ██   ██ ██
# ██   ███ ██████  ██    ██     ██      ██    ██ ██   ██ █████
# ██    ██ ██      ██    ██     ██      ██    ██ ██   ██ ██
#  ██████  ██       ██████       ██████  ██████  ██████  ███████
#
# This can handle specialiced models with up to 48 bit search space
# It optimizes a LOT.


class TmpVariableGenerator:

	def __init__(self):
		self.lookup = {}
		self.code = []
		self.counter = 0

	def __getitem__(self, kernel):
		if kernel in self.lookup:
			return self.lookup[kernel]
		self.lookup[kernel] = f"tmp_{self.counter}"
		self.code.append(f"int {self.lookup[kernel]} = {kernel};")
		self.counter += 1
		return self.lookup[kernel]


ENUMERATOR_KEY = 'e'


def is_model_supported_on_gpu(model):

	implemented = [rda.power.model.HW, rda.power.model.HD, rda.power.model.HWSBOX, rda.power.model.Public]

	def is_implemented(component):
		return any([isinstance(component, t) for t in implemented])

	return all([is_implemented(c) for c in model.components])


def replace_secret_word_with_enumerator(mc):
	ENUMERATOR = rda.power.model.Secret(ENUMERATOR_KEY)

	if isinstance(mc, rda.power.model.Public):
		return mc, None

	has_sw = [len(child.secret_words()) > 0 for child in mc.children]
	has_pw = [len(child.public_words()) > 0 for child in mc.children]

	has_both = [sw and pw for sw, pw in zip(has_sw, has_pw)]

	assert (not any(has_both))
	# currently only up to one secret enumeration is supported
	assert (sum(has_sw) <= 1)

	new_children = [ENUMERATOR if is_sw else child for is_sw, child in zip(has_sw, mc.children)]

	# construct the adapted model_component
	new_mc = type(mc)(*new_children)
	new_mc.coefficient = mc.coefficient

	# extract the numeration "formula"
	enumeration_formula = mc.children[has_sw.index(True)] if any(has_sw) else None

	return new_mc, enumeration_formula


def adapt_model_for_cpa(model):

	model_components = []
	enumeration_formulas = []

	for model_component in model.components:
		mc, ef = replace_secret_word_with_enumerator(model_component)
		model_components.append(mc)
		enumeration_formulas.append(ef)
		#print(f'{mc!s} -> {ef!s}')

	return rda.power.model.Model(model_components), enumeration_formulas


def compute_cov_and_var(df, word_candidates, measurements_column, model, normalization: CPA_Normalization):
	'''
		This functions computes optimized structures for the pearson r coefficient:

		r = Cov(X, Y) / sqrt(Var(X)*Var(Y)) = Cov(X, Y) / (std(X)*std(Y))

		we optimize the following structure:

		a = Cov(X, Y) / std(Y) 
		b = Var(X)

		r = a / sqrt(b)

		X -> h(sv): the hypothesis depending on the enumerated secret values
		Y -> m: the measurements

		the hypothesis is constructed over smaller parts where each part evaluates only a subspace 
		of the enumeration space

		h(sv)[i] = h_1(sv1')[i] + h_2(sv2')[i] + ... + h_k(svk')[i]

		a = Cov(X,Y) / std(Y) = 1/(n-1) * sum ((h(sv)[i] - mean(h(sv)) * (m[i] - mean(m)) / std(m)

		we can precompute "(m[i] - mean(m)) / std(m)" by using the zscore of m

		z[i] = zscore(m)[i]

		we define the mean subtracted hypothesis hn(sv)[i] = h(sv)[i] - mean(h(sv))

		S = 1/(n-1)

		a = Cov(X,Y) / std(Y) = S * sum (hn(sv)[i] * z[i])

		we can expand a for each partial hypothesis:

		a = S * sum (hn(sv)[i] * z[i]) = S * sum(hn_1(sv1')[i] * z[i] + hn_2(sv2')[i] * z[i] + ... + hn_k(svk')[i] * z[i])
		
		we precompute: hn_1(sv1')[i] * z[i] * S which allows us to represent a as a sum of array accesses based on the secret parts.

		b = Var(X) = 1 / (n-1) * sum((x_i - x_bar)^2) = S * sum( hn(sv)[i]^2 ) = 
		b = S * sum(  (hn_1(sv1')[i] + hn_2(sv2')[i]  + ... + hn_k(svk')[i] )  ^2 )



		X is the hypothesis of N samples -> we can compute all cnadidates in matrix form:

		X = [n_candidates, N]

		X - 

	
	'''
	dtype = np.float64

	mempool = cp.get_default_memory_pool()

	# TODO: dynamically compute this
	n_step = 3_000_000

	pws = model.public_words()
	#pw_index_lookup = {pw: i for i, pw in enumerate(pws)}

	# normalize the measurements -> variance formula
	measurements = scipy.stats.zscore(df[measurements_column]).astype(dtype)
	#pw_values = df[pws].values

	# shape the arrays for faster access on the gpu
	pw_values = {pw: df[pw].values[np.newaxis, :] for pw in pws}

	#pw_values = pw_values.T[:, np.newaxis, :]
	word_candidates = word_candidates[:, np.newaxis]

	# disable cupy since they have numerical instabilities
	xp = cp

	# copy to gpu if used
	if xp == cp:
		word_candidates = cp.asarray(word_candidates)
		measurements = cp.asarray(measurements)
		pw_values = {k: cp.asarray(v) for k, v in pw_values.items()}
		#pw_values = cp.asarray(pw_values)

	# candidates = xp.arange(2**ctx_power.secret_word_ctx.bits, dtype=ctx_power.secret_word_ctx.dtype)

	# custom data getter so we can change the axis of the data
	# this allows the broadcasts within the model to do the correct
	# thing when secret and public values are combined
	# futhermore we slice the input candidates to not exceed memory
	@dataclass(frozen=True)
	class DataAdapter:
		n_start: int

		def __getitem__(self, key):
			# check if a secret value -> enumerate
			if key == ENUMERATOR_KEY:
				return word_candidates

			# or a public value
			elif key in pws:
				return pw_values[key][:, self.n_start:self.n_start + n_step]
				#return pw_values[np.newaxis, self.n_start:self.n_start + n_step, pw_index_lookup[key]]

				x = pw_values[pw_index_lookup[key], :]
				return x[np.newaxis, self.n_start:self.n_start + n_step]

			# huh? error case
			else:
				raise KeyError(
				    f'got key which is not in the publics or an enumerator: {key} - is the model implemented correctly?'
				)

	# temporaries
	cov_tmp = xp.zeros((word_candidates.shape[0],), dtype=dtype)
	var_tmp = xp.zeros((word_candidates.shape[0], word_candidates.shape[0]), dtype=dtype)

	def triangle_elements(N):
		return (N * (N + 1)) // 2

	# zero everything
	cov = xp.zeros((len(model.components), word_candidates.shape[0]), dtype=dtype)
	var = xp.zeros((triangle_elements(len(model.components)), word_candidates.shape[0], word_candidates.shape[0]),
	               dtype=dtype)

	idx_lookup = {
	    (id(c_i), id(c_j)): i
	    for i, (c_i, c_j) in enumerate(itertools.combinations_with_replacement(model.components, 2))
	}

	def eval_hypothesis_without_mean(model_component, data, dtype, n_broadcast):
		# evaluate the hypothesis
		hypothesis = model_component.eval(data).astype(dtype)

		# remove the mean inplace
		hypothesis -= xp.mean(hypothesis, axis=1, keepdims=True, dtype=dtype)

		# for constant model components (like hw(public) or hw(secret)) the shape needs to be broadcasted
		expected_shape = (word_candidates.shape[0], n_broadcast)

		if hypothesis.shape != expected_shape:
			hypothesis = xp.broadcast_to(hypothesis, expected_shape)

		# manually free the mean array
		mempool.free_all_blocks()

		# return the new_hypothesis and hope the memory is not copied
		return hypothesis

	def update_covariance(idx, hypothesis, observations):
		# compute covariance and add it to the iterative result
		xp.dot(hypothesis, observations, out=cov_tmp)
		cov[idx, :] += cov_tmp

	def update_variance(model_i, model_j, hypothesis_i, hypothesis_j):
		idx = idx_lookup[(id(model_i), id(model_j))]
		# compute variance and add it to the iterative result
		xp.dot(hypothesis_i, hypothesis_j.T, out=var_tmp)
		var[idx, :, :] += var_tmp

	# iterate over the measurement slices
	for n_start in tqdm(range(0, measurements.shape[0], n_step), desc="Prg", leave=False):

		# generate a data adapter with the current slice
		da = DataAdapter(n_start)

		# go over each component in the model
		for i, model_i in enumerate(tqdm(model.components[:], desc="COV", leave=False)):

			n_broadcast = measurements[n_start:n_start + n_step].shape[0]

			hypothesis_i = eval_hypothesis_without_mean(model_i, da, dtype, n_broadcast)

			update_covariance(i, hypothesis_i, measurements[n_start:n_start + n_step])

			# skip the variance computation if not requested
			if normalization in [CPA_Normalization.INLINE, CPA_Normalization.POST]:

				update_variance(model_i, model_i, hypothesis_i, hypothesis_i)

				for model_j in tqdm(model.components[i + 1:], desc="VAR", leave=False):
					hypothesis_j = eval_hypothesis_without_mean(model_j, da, dtype, n_broadcast)

					update_variance(model_i, model_j, hypothesis_i, hypothesis_j)

					# manually free unused memory
					del hypothesis_j
					mempool.free_all_blocks()

			# manually free unused memory
			del hypothesis_i
			mempool.free_all_blocks()

	# normalize covariance and variance
	cov /= measurements.shape[0] - 1
	var /= measurements.shape[0] - 1

	for model_i, model_j in itertools.combinations_with_replacement(model.components, 2):
		idx = idx_lookup[(id(model_i), id(model_j))]
		scale = 1 if id(model_i) == id(model_j) else 2
		var[idx, :, :] *= scale

	return cov, var


FMT_NEWLINE = "\n\t\t"
FMT_PLUS_AND_NEWLINE = " + " + FMT_NEWLINE


def generate_cov_and_variance_code(enumeration_formulas, lookup_type='index'):

	# get state for the temporary variables
	tmp_variables = TmpVariableGenerator()

	def get_tmp_var_name_from_ef(ef):
		# if we get passed a None as ef this means
		# that the model does not have an enumeration
		# ie it is constant all the time and we simply index
		# the first element -> TODO for later
		return ef.get_kernel(tmp_variables) if ef else '0'

	# generate covariance variables
	covariances = []
	for idx, ef in enumerate(enumeration_formulas):
		t0 = get_tmp_var_name_from_ef(ef)
		covariances.append(rda.optimized.utils.lookup("float", "covariances", idx, t0, lookup_type=lookup_type))

	# generate variance variables
	variances = []
	for idx, (ef_i, ef_j) in enumerate(itertools.combinations_with_replacement(enumeration_formulas, 2)):
		t0 = get_tmp_var_name_from_ef(ef_i)
		t1 = get_tmp_var_name_from_ef(ef_j)
		variances.append(rda.optimized.utils.lookup("float", "variances", idx, t0, t1, lookup_type=lookup_type))

	# add all covariances
	covariance_code = FMT_PLUS_AND_NEWLINE.join(covariances)
	variance_code = FMT_PLUS_AND_NEWLINE.join(variances)
	temporaries_code = FMT_NEWLINE.join(tmp_variables.code)

	return covariance_code, f'sqrt({variance_code})', temporaries_code


def generate_secret_word_from_index_code(word_name, byte_index):
	return f'unsigned char {word_name} = i >> {byte_index*8};'


def generate_cpa_kernel(model, enumeration_formulas, covariances, variances, n_candidates, n_candidate_storage,
                        normalization: CPA_Normalization):

	# generate code for covariance and variance lookup
	cov_code, var_code, tmp_code = generate_cov_and_variance_code(enumeration_formulas, lookup_type='index')

	match normalization:
		case CPA_Normalization.NONE:
			corrcoef_code = f'({cov_code})'
		case CPA_Normalization.POST:
			corrcoef_code = f'({cov_code})'
		case CPA_Normalization.INLINE:
			corrcoef_code = f'({cov_code})/({var_code})'

	secret_combinations = FMT_NEWLINE.join(
	    [generate_secret_word_from_index_code(w, i) for i, w in enumerate(model.secret_words())])

	c_dtype = 'double'
	cp_dtype = np.float64

	code = rf"""
		#include <curand.h>
		#include <curand_kernel.h>
		#include <cuda_fp16.h>

		__device__ unsigned char sbox[256];
		//__device__ unsigned char sbox_inv[256];

		__device__ {c_dtype} covariances[{covariances.shape[0]}][{covariances.shape[1]}];
		__device__ {c_dtype} variances[{variances.shape[0]}][{variances.shape[1]}][{variances.shape[2]}];

		#define N ({n_candidate_storage})

		__device__ double corrcoefs[N];
		__device__ unsigned long long candidates[N];

		extern "C" __global__ void cpa_sampled() {{
		
			
			unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;

			curandState state;
			curand_init(1234, tid, 0, &state);

			for (unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x) {{

				unsigned long long i = ((unsigned long long)(curand(&state)) << 32) | curand(&state);

				{secret_combinations}
				{tmp_code}

				double r = {corrcoef_code};

				corrcoefs[tid] = r;
				candidates[tid] = i;
			}}
		}}

		

		extern "C" __global__ void cpa(double threshold, unsigned long long *counter) {{
		
		
			for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < {n_candidates}; i += blockDim.x * gridDim.x) {{
			
				{secret_combinations}
				{tmp_code}

				double r = {corrcoef_code};

				if (r >= threshold) {{
					unsigned long long idx = atomicAdd(&counter[0], 1);
					if (idx < N) {{
						candidates[idx] = i;
						corrcoefs[idx] = r;
					}}
				}}
			}}
		}}

		extern "C" __global__ void normalize_cpa_results() {{
	

			for (unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x) {{

				unsigned long long i = candidates[tid];

				{secret_combinations}
				{tmp_code}

				corrcoefs[tid] = corrcoefs[tid] / ({var_code});
			}}
		}}

	"""

	module = cp.RawModule(code=code, options=("--std=c++17",))

	# get the function handles
	kernel_cpa = module.get_function("cpa")
	kernel_cpa_sampled = module.get_function("cpa_sampled")
	kernel_normalize_cpa_results = module.get_function("normalize_cpa_results")

	# get memory handles to the sbox and inverse sbox
	sbox = cp.ndarray((256,), cp.uint8, module.get_global("sbox"))
	#sbox_inv = cp.ndarray((256,), cp.uint8, module.get_global("sbox_inv"))

	# copy the sbox and inverse sbox to the shaders
	sbox[:] = cp.asarray(rda.optimized.functions.lookup_sbox)
	#sbox_inv[:] = cp.asarray(rda.optimized.functions.lookup_sbox_inv)

	# get covariances
	kernel_covariances = cp.ndarray(covariances.shape, cp_dtype, module.get_global("covariances"))
	kernel_covariances[:] = cp.asarray(covariances.astype(cp_dtype))

	# get variances
	kernel_variances = cp.ndarray(variances.shape, cp_dtype, module.get_global("variances"))
	kernel_variances[:] = cp.asarray(variances.astype(cp_dtype))

	# handle to candidates and correlations
	candidates = cp.ndarray(n_candidate_storage, cp.uint64, module.get_global("candidates"))
	corrcoefs = cp.ndarray(n_candidate_storage, cp.float64, module.get_global("corrcoefs"))

	def cpa(rho_threshold, counter):
		kernel_cpa((512,), (256,), (rho_threshold, counter))
		if counter > n_candidate_storage:
			print("ERROR COULD NOT STORE ALL CANDIDATES")

	def cpa_sampled():
		kernel_cpa_sampled((1024,), (256,), ())

	def normalize_cpa_results():
		# only perform the normalization of the results if the results was not already normalized
		if normalization == CPA_Normalization.POST:
			kernel_normalize_cpa_results((1024,), (256,), ())

	@dataclass(frozen=True)
	class KernelData:
		cpa: None
		cpa_sampled: None
		normalize_cpa_results: None
		corrcoefs: None
		candidates: None

	return KernelData(cpa, cpa_sampled, normalize_cpa_results, corrcoefs, candidates)


def estimate_rank(df, measurements_column, model, normalization, sampled_corrcoef, n_overall_candidates):
	# evaluate the model directly to get the hypothesis for the correct secrets
	H = model.eval(df)

	# get the corrcoef of the ground truth
	# if no normalization is requested we need to revert the built in normalization of scipy
	if normalization == CPA_Normalization.NONE:
		gt_corrcoef = scipy.stats.pearsonr(H, df[measurements_column]).statistic * np.std(H, ddof=1)
	else:
		gt_corrcoef = scipy.stats.pearsonr(H, df[measurements_column]).statistic

	# find the insert index of the ground truth corrcoef in the distribution
	insert_index = cp.searchsorted(
	    sampled_corrcoef,
	    cp.asarray(gt_corrcoef),
	    sorter=cp.argsort(sampled_corrcoef),
	)

	# scale the location based on the ratio of the sample size and the overall population size
	rank = (1 - np.float64(insert_index.get()) / sampled_corrcoef.shape[0]) * n_overall_candidates

	# extract the correct secret values of the gt
	secret_values = {w: np.array([df[w].iloc[0]]) for w in model.secret_words()}

	# generate a result data frame
	results = pd.DataFrame(secret_values)

	# append the corrcoefs
	results["rho"] = gt_corrcoef
	results["rank"] = rank

	return results


def compute_cpa_gpu(df, ctx_power, measurements_column, model, normalization, profile=False, only_estimate_rank=False):
	if not is_model_supported_on_gpu(model):
		raise NotImplementedError(f'the model is not implemented to run with cuda: {model}')

	# number of reported key candidates
	n_key_candidates = 200_000_000

	# convert the normalization string
	normalization = CPA_Normalization[normalization.upper()]

	# replace the secret values in the model with a cpa capable iterator
	adapted_model, enumeration_formulas = adapt_model_for_cpa(model)

	# the iteration space for each individual secret word
	n_word_candidates = 2**ctx_power.secret_word_ctx.bits

	# each of the possible word values
	word_candidates = np.arange(n_word_candidates, dtype=ctx_power.secret_word_ctx.dtype)

	# the total iteration space of the cpa
	n_overall_candidates = n_word_candidates**len(model.secret_words())

	if ctx_power.secret_word_ctx.bits * len(model.secret_words()) >= 64:
		raise NotImplementedError(f'model exceeds the iteration space of 64 bits: {model}')

	# get the covariance and potentially the variance of the provided model
	cov, var = compute_cov_and_var(df, word_candidates, measurements_column, adapted_model, normalization)

	# overprovision the gpu storage
	n_candidate_storage = min(round(n_key_candidates * 1.5), n_overall_candidates)

	# generate the cuda kernels
	kernel = generate_cpa_kernel(
	    model,
	    enumeration_formulas,
	    cov,
	    var,
	    n_overall_candidates,
	    n_candidate_storage,
	    normalization,
	)

	# sample the correlation distribution
	kernel.cpa_sampled()

	# check if we want to only perform the rank estimation
	if only_estimate_rank:

		# normalize the sampled distribution if requested
		if normalization == CPA_Normalization.POST or normalization == CPA_Normalization.INLINE:
			kernel.normalize_cpa_results()

		result = estimate_rank(df, measurements_column, model, normalization, kernel.corrcoefs, n_overall_candidates)

		# free kernel
		del kernel
		# make sure everything is freed
		cp.get_default_memory_pool().free_all_blocks()

		return result

	# otherwise we perform the full CPA

	if n_key_candidates > n_overall_candidates:
		# if we have more space than we need we can record all candidates therefore no threshold
		rho_threshold = np.float64(-2)  # choose somthing which is imposible rho is withing -1 1

	else:
		# if limited space - we select a threshold that should give us the expected number of candidates

		# compute the estimated percentile to get n_key_candidates
		percentile = np.clip(100.0 - (n_key_candidates / n_overall_candidates) * 100, 0, 100.0)

		# get the threshold for the cpa
		rho_threshold = np.float64(cp.percentile(kernel.corrcoefs, percentile).get())

	# reset the kernel's storage arrays
	kernel.candidates[:] = 0
	kernel.corrcoefs[:] = 0

	# counter for sampling
	counter = cp.zeros((1,), dtype=np.uint64)
	counter[0] = 0

	if profile:
		start_gpu = cp.cuda.Event()
		end_gpu = cp.cuda.Event()
		start_gpu.record()
		start_cpu = time.perf_counter()

	kernel.cpa(rho_threshold, counter)
	kernel.normalize_cpa_results()

	if profile:
		end_cpu = time.perf_counter()
		end_gpu.record()
		end_gpu.synchronize()

		t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
		t_cpu = end_cpu - start_cpu

		print(f'GPU: {t_gpu}ms CPU: {t_cpu}s - CPA found {counter[0]} candidates with a threshold of {rho_threshold}')

	# get the candidates up to the counter
	corrcoefs = kernel.corrcoefs[:counter[0]]
	candidates = kernel.candidates[:counter[0]]

	# sort by the correlation coefficient and drop everything below
	# the number of key candidates
	index = cp.argsort(corrcoefs)[-n_key_candidates:]

	# invert the sort order
	corrcoefs = corrcoefs[index][::-1]
	candidates = candidates[index][::-1]

	# copy the data back to the cpu
	corrcoefs = cp.asnumpy(corrcoefs)
	candidates = cp.asnumpy(candidates)

	# extract
	# TODO: generalize beyond 8 bit
	secret_values = {w: (candidates >> (8 * i)).astype(np.uint8) for i, w in enumerate(model.secret_words())}

	# generate a result data frame
	results = pd.DataFrame(secret_values)

	# append the corrcoefs
	results['rho'] = corrcoefs

	# get the ground truth
	ground_truth = df[model.secret_words()].iloc[0].values

	# get the rank if contained
	rank = np.where(np.all(results[model.secret_words()].values == ground_truth, axis=1))[0]

	# we couldn't find the rank -> lets estimate
	if len(rank) == 0:
		# sample the correlation distribution
		kernel.cpa_sampled()

		# normalize the sampled distribution if requested
		if normalization == CPA_Normalization.POST or normalization == CPA_Normalization.INLINE:
			kernel.normalize_cpa_results()

		estimate = estimate_rank(
		    df,
		    measurements_column,
		    model,
		    normalization,
		    kernel.corrcoefs,
		    n_overall_candidates,
		).iloc[0]

		# if we couldn't find the correct value it has to be not in your candidate list therefore
		# the rank mus be larger as your candidate list

		results['rank'] = max(estimate['rank'], corrcoefs.shape[0])
		results['rho_gt'] = estimate['rho']
		results['est'] = True
	else:
		results['rank'] = rank[0]
		results['rho_gt'] = results['rho'].iloc[rank[0]]
		results['est'] = False

	# free kernel
	del kernel
	# make sure everything is freed
	cp.get_default_memory_pool().free_all_blocks()

	return results
