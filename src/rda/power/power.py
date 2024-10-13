import pandas as pd
import numpy as np
import cupy as cp
import contextlib

import warnings
import scipy.stats

from sklearn.linear_model import LinearRegression, HuberRegressor, SGDRegressor

from tqdm import tqdm
from typing import List, Union, Any, Tuple
from warnings import catch_warnings, simplefilter
from dataclasses import dataclass, field

import rda.utils.utils

import rda.power.model
import rda.power.cpa

import rda.optimized.utils
import rda.optimized.pearsonr
import rda.optimized.expander
import rda.aes.aes

from rda.cli.command import GroupedData, UngroupedData, command, tuple_list

import rda.utils.colors as colors

from multiprocessing import Pool

import rda.optimized.functions

from enum import Enum


class FilterUnusedWords(Enum):
	NONE = 1
	ALL = 2
	GUESS = 3


@dataclass
class WordContext:
	'''class to describe the word width and the dtype representing the words'''

	bits: int = 0
	count: int = 0
	dtype: np.dtype = np.uint8
	expander: rda.optimized.expander.SimpleExpander = (None)  # rda.optimized.expander.ExtendedExpander


@dataclass
class PowerContext:
	'''class to keep track of the power related meta data'''

	# Defines the sizes of an analysis word
	secret_word_ctx: WordContext = None
	public_word_ctx: WordContext = None

	# Victim and Guess words
	vs: List[rda.power.model.Secret] = field(default_factory=list)
	gs: List[rda.power.model.Public] = field(default_factory=list)

	# string rep for pandas
	vs_str: List[str] = field(default_factory=list)
	gs_str: List[str] = field(default_factory=list)

	# the model components
	models: List[rda.power.model.Model] = field(default_factory=list)


power_diff_columns = [
    'Power',
    'PowerPP0',
    'PowerDRAM',
    'Energy',
    'EnergyPP0',
    'EnergyDRAM',
    'Freq',
    'Ticks',
    'Volt',
    'inst_a',
    'inst_b',
    'temp_a',
    'temp_b',
    'Temp',
    'hd_vg',
    'hw_g',
    'hw_state_g',
    'hw_ct_g',
]


@command('power_sim_aes_sbox', grouped=False)
def power_sim_aes_sbox(data: UngroupedData, number_samples: int, train_test_split=0.3, snr=0.01):
	'''simulate power related data'''

	N_train = int(number_samples * train_test_split)

	np.random.seed(seed=123)

	fixed_key = np.array(
	    [
	        0x49,
	        0x43,
	        0xCA,
	        0x87,
	        0x22,
	        0x98,
	        0x77,
	        0x65,
	        0x45,
	        0x34,
	        0x32,
	        0x12,
	        0x19,
	        0xa1,
	        0x42,
	        0x3C,
	    ],
	    dtype=np.uint8,
	)

	random_keys = np.random.randint(0, high=256, size=(N_train, 16), dtype=np.uint8)

	# set half of the keys to the fixed key
	master_keys = np.zeros((number_samples, 16), dtype=np.uint8)

	master_keys[:N_train, :] = random_keys
	master_keys[N_train:, :] = fixed_key

	K = master_keys

	# plaintextes

	P = np.random.randint(0, high=256, size=(number_samples, 16), dtype=np.uint8)

	def simulate_energy(P, K):

		sbox = rda.optimized.functions.sbox(K[:, 0:1] ^ P[:, 0:1])
		est = np.sum(rda.optimized.functions.hw(sbox), axis=1).astype(np.float64)
		est += np.random.normal(np.mean(est), np.sqrt(np.var(est) * 1.0 / snr), est.shape)

		return est

	byte_guess = []
	byte_value = []

	for i in range(number_samples):
		byte_guess.append(P[i, :].tobytes())
		byte_value.append(K[i, :].tobytes())

	df = pd.DataFrame()

	df['Guess'] = byte_guess
	df['Value'] = byte_value
	df['Energy'] = simulate_energy(P, K)
	df['EnergyPP0'] = df['Energy']
	df['EnergyDRAM'] = df['Energy']
	df['Ticks'] = 1
	df['APerf'] = 1
	df['Mperf'] = 1
	df['PState'] = 1
	df['Temp'] = 1
	df['temp_a'] = 1
	df['temp_b'] = 1
	df['la'] = 1
	df['Volt'] = 1

	df['ERep'] = 0
	df['SRep'] = 0

	df['Exp'] = 'train'
	df['Exp'].iloc[N_train:] = 'test'

	return data.ctx.undo_grouping([df])


#####################################
#
# COMMANDS
#
@command('power_init')
def power_init_df(
    data: GroupedData,
    machine: str = '',
    no_diff: bool = False,
    no_expand: bool = False,
    bits: int = 8,
    wstride: int = 1,
    wcount: int = -1,
    filter_unused: str = 'NONE',
    only_diff: bool = False,
):
	'''initialize additional power related columns and meta data'''

	df = data.df
	ctx = data.ctx

	if 'Value' not in df or 'Guess' not in df:
		raise RuntimeError('data frame does not have columns Value or Guess')

	# fix the overflow in the only 16 bit long erep and srep fields
	# TODO: overall fix  this - how to handle legacy? :/
	df.ERep += (df.ERep.astype(np.int64).diff() < 0).cumsum() * 2**16
	df['EERep'] = (df.ERep % 100) * 50 + df.SRep

	# set the data types manually - this saves a lot of trouble later
	df = df.astype({'Value': '|S192', 'Guess': '|S192'})

	print('- converting units')
	df = power_convert_units(df, machine)

	print('- computing additional fields')
	df = power_compute_power_and_freq(df)

	print('- creating power context')
	ctx.module_data['power'] = create_power_ctx(df, bits, wstride, wcount)

	if not no_expand:
		print('- expanding intermediates')
		df = power_expand_intermediates(df, ctx.module_data['power'], FilterUnusedWords[filter_unused.upper()])

	if not no_diff:
		print('- applying differential measurement')
		df = power_new_to_diff(df, only_diff)

	print(f'Columns: {" ".join(df.columns)}')

	return df


def power_convert_units(df, machine):
	machine_scales = {
	    'mlab07': {
	        'Energy': 6.103515625e-05,
	        'EnergyPP0': 6.103515625e-05,
	        'Ticks': 2.7027027e-10,
	        'APerf': 3700,
	        'PState': 100,
	        'Volt': 1 / 8192.0
	    },
	    'ulab07': {
	        'Energy': 6.103515625e-05,
	        'EnergyPP0': 6.103515625e-05,
	        'Ticks': 4.1666667e-10,
	        'APerf': 2400,
	        'PState': 100,
	        'Volt': 1 / 8192.0
	    },
	    'lab10': {
	        'Ticks': 2.7777778e-10,
	        'Energy': 6.103515625e-05
	    },
	    'lab06': {
	        'Ticks': 2.5e-10
	    },
	    'lap': {
	        'Energy': 6.103515625e-05,
	        'EnergyPP0': 6.103515625e-05,
	        'Ticks': 1 / (1.6 * 10**9)
	    }
	}

	if machine not in machine_scales:
		return df

	for column, scale in machine_scales[machine].items():
		if column in df:
			df[column] *= scale

	print('-- number of seconds per sample:')

	if 'Ticks' in df and 'Exp' in df:
		print(df.groupby('Exp').Ticks.mean())

	return df


def power_compute_power_and_freq(df):

	for postfix in ['', 'PP0']:
		col_energy = 'Energy' + postfix
		col_power = 'Power' + postfix

		if col_energy in df and 'Ticks' in df:
			print(f'-- computing {col_power}')
			df[col_power] = df[col_energy] / df['Ticks']

	if 'APerf' in df and 'Mperf' in df:
		print(f'-- computing Freq')
		df['Freq'] = df['APerf'] / df['Mperf']

	if False and 'Flushes' in df:
		print(f'-- computing Flush Freq')
		df['flush_freq'] = df['Flushes'] / df['Ticks']

	return df


def create_power_ctx(df, word_bits: int, word_stride: int, word_count: int, infer=False):

	if infer:
		word_stride = 1
		gs = [c for c in df.columns if c.startswith('g') and len(c) == 4]
		vs = [c for c in df.columns if c.startswith('v') and len(c) == 4]
		vs_word_count = len(vs)
		gs_word_count = len(gs)
	else:
		vs_word_count = word_count
		gs_word_count = word_count

	def gen_word_ctx(column, word_count):
		if word_bits > 0 and word_count >= 0:
			data_bits = word_bits * word_count
		else:
			data_bits = df[column].dtype.itemsize * 8

		assert (data_bits % word_bits == 0)

		wc = word_count if word_count != -1 else (data_bits // word_bits) // word_stride
		wc1 = word_count if word_count != -1 else data_bits // word_bits

		return WordContext(
		    word_bits,
		    wc,
		    rda.optimized.utils.dtype_from_bit(word_bits),
		    rda.optimized.expander.create_expander(word_bits, wc1, word_stride),
		)

	power_ctx = PowerContext()
	power_ctx.secret_word_ctx = gen_word_ctx('Value', vs_word_count)
	power_ctx.public_word_ctx = gen_word_ctx('Guess', gs_word_count)

	# the words available in the data frame
	if infer:
		power_ctx.vs = [rda.power.model.Secret(v) for v in vs]
		power_ctx.gs = [rda.power.model.Public(g) for g in gs]
	else:
		power_ctx.vs = [rda.power.model.Secret(f'v{i:03}') for i in range(power_ctx.secret_word_ctx.count)]
		power_ctx.gs = [rda.power.model.Public(f'g{i:03}') for i in range(power_ctx.public_word_ctx.count)]

	# pandas likes string columns
	power_ctx.vs_str = list(map(str, power_ctx.vs))
	power_ctx.gs_str = list(map(str, power_ctx.gs))

	return power_ctx


def power_expand_intermediates(df, meta: PowerContext, filter_unused: FilterUnusedWords):

	print('-- expanding value')
	vs = meta.secret_word_ctx.expander(df['Value'].values)

	print('-- expanding guess')
	gs = meta.public_word_ctx.expander(df['Guess'].values)

	#vs = np.zeros(gs.shape, dtype=gs.dtype)

	print('-- computing hw state guess')

	#if gs.flags['C_CONTIGUOUS']:
	#	hw_state_g, hw_ct_g = rda.aes.aes.compute_key_depending_hws(gs)
	#	df_hw_state_g = pd.DataFrame(hw_state_g, columns=['hw_state_g'], index=df.index, copy=False)
	#	df_hw_ct_g = pd.DataFrame(hw_ct_g, columns=['hw_ct_g'], index=df.index, copy=False)
	#else:
	#	print('warning cannot compute hw states due to striding')
	#	df_hw_state_g = pd.DataFrame()
	#	df_hw_ct_g = pd.DataFrame()

	if filter_unused != FilterUnusedWords.NONE:
		print('-- filtering unused intermediates')

		if filter_unused == FilterUnusedWords.ALL:
			vs_mask = np.where(vs.any(axis=0))[0]
			gs_mask = np.where(gs.any(axis=0))[0]
		elif filter_unused == FilterUnusedWords.GUESS:
			vs_mask = np.arange(vs.shape[1])
			gs_mask = np.where(gs.any(axis=0))[0]
		else:
			assert (False)

		mask = np.array(list(set(vs_mask) & set(gs_mask)))

		print('-- computing hd')
		#hd_vg = rda.optimized.functions.hd(gs[:, mask], vs[:, mask]).sum(axis=1, dtype=np.int64)
		#hd_vg_p2 = rda.optimized.functions.hd(gs[:, mask], vs[:, mask + 32]).sum(axis=1, dtype=np.int64)
		#hd_vg_p1 = rda.optimized.functions.hd(gs[:, mask], vs[:, mask + 16]).sum(axis=1, dtype=np.int64)
		#hd_vg_m1 = rda.optimized.functions.hd(gs[:, mask], vs[:, mask - 16]).sum(axis=1, dtype=np.int64)
		#hd_vg_m2 = rda.optimized.functions.hd(gs[:, mask], vs[:, mask - 32]).sum(axis=1, dtype=np.int64)

		vs = vs[:, vs_mask]
		gs = gs[:, gs_mask]

		meta.vs = [meta.vs[i] for i in vs_mask]
		meta.gs = [meta.gs[i] for i in gs_mask]

		meta.vs_str = [meta.vs_str[i] for i in vs_mask]
		meta.gs_str = [meta.gs_str[i] for i in gs_mask]

	else:
		print('-- computing hd')
		hd_vg = rda.optimized.functions.hd(gs, vs).sum(axis=1, dtype=np.int64)

	print('-- computing hw_g')
	hw_g = rda.optimized.functions.hw(gs).sum(axis=1, dtype=np.int64)

	print('-- creating data frame')
	df_vs = pd.DataFrame(vs, columns=meta.vs_str, index=df.index, copy=False)
	df_gs = pd.DataFrame(gs, columns=meta.gs_str, index=df.index, copy=False)

	#df_hd_vg = pd.DataFrame(hd_vg, columns=['hd_vg'], index=df.index, copy=False)
	df_hw_g = pd.DataFrame(hw_g, columns=['hw_g'], index=df.index, copy=False)

	df.drop(['Guess', 'Value'], inplace=True, axis=1)

	print('-- appending data frame')
	df = pd.concat([df, df_vs, df_gs, df_hw_g], axis=1, copy=False)

	return df


def power_new_to_diff(df, only_diff=False):
	''' The new format reports no differential power so compute the fields manually by using the adjacent fields'''

	columns = [c for c in power_diff_columns if c in df.columns]

	# convert the columns to an unsigned type for the diff
	# use only the first row to save compute
	cvt_dtypes = df.loc[0:0, columns].apply(lambda x: pd.to_numeric(x, downcast='signed')).dtypes

	# update the dtypes
	df = df.astype({name: dtype for name, dtype in cvt_dtypes.items()}, copy=False)

	# split into real and inverse samples
	df_r = df.loc[0::2, :]
	df_i = df.loc[1::2, columns].set_index(df_r.index)

	if only_diff:
		df_r.loc[:, columns] = df_r[columns].sub(df_i)
		return df_r

	df_d = df_r[columns].sub(df_i)

	# prefix the columns
	df_r = df_r.rename(columns=dict({c: 'R' + c for c in columns}))
	df_i = df_i.rename(columns=dict({c: 'I' + c for c in columns}))
	df_d = df_d.rename(columns=dict({c: 'D' + c for c in columns}))

	# return the overall data frame
	return pd.concat([df_r, df_i, df_d], axis=1, ignore_index=False, copy=False)


@command('power_ctx', grouped=False)
def create_power_ctx_cmd(data: GroupedData, bits: int = 8, wstride: int = 1, wcount: int = -1, infer=False):
	data.ctx.module_data['power'] = create_power_ctx(data.dfs[0], bits, wstride, wcount, infer)
	return data.dfs


@command('power_models')
def power_set_models(data: GroupedData, *models):
	'''set the power related models'''

	ctx_power = data.ctx.module_data['power']
	ctx_power.models = []

	print('models:')
	for x in models:
		m = rda.power.model.model_from_string(x, data.df.columns, ctx_power.vs, ctx_power.gs)
		ctx_power.models.append(m)
		print(m.describe())

	return data.df


@command('power_eval_model')
def power_eval_model(data: GroupedData, model):

	ctx_power = data.ctx.module_data['power']

	m = rda.power.model.model_from_string(model, data.df.columns, ctx_power.vs, ctx_power.gs)
	data.df[str(model)] = m.eval(data.df)

	return data.df


def power_cross_correlation(x, y):
	x = scipy.stats.zscore(x.astype(np.float64))
	y = scipy.stats.zscore(y.astype(np.float64))

	rhos = scipy.signal.correlate(x, y, mode='same')
	rhos /= rhos.max()
	lags = scipy.signal.correlation_lags(x.shape[0], y.shape[0], mode='same')

	return pd.DataFrame({'rho': rhos, 'lags': lags}, index=lags)


@command('power_xcor')
def power_xcor(data: GroupedData):
	x = data.df.IPowerPP0.values
	y = data.df.RPowerPP0.values
	return power_cross_correlation(x, y)


@command('power_acor')
def power_xcor(data: GroupedData):
	x = data.df.DPowerPP0.values
	return power_cross_correlation(x, x)


@command('power_cov')
def power_cov(data: GroupedData):
	df = data.df
	ctx = data.ctx
	'''print the covariances and corrcoefs of the R and I energy and power columns'''
	ctx.print_group_header(df)

	energy_cov_matrix = df[['REnergyPP0', 'IEnergyPP0']].cov()
	power_cov_matrix = df[['RPowerPP0', 'IPowerPP0']].cov()

	energy_corrcoef = scipy.stats.pearsonr(df.REnergyPP0.iloc[1:], df.REnergyPP0.shift(1).iloc[1:])
	power_corrcoef = scipy.stats.pearsonr(df.RPowerPP0.iloc[1:], df.RPowerPP0.shift(1).iloc[1:])

	x = pd.DataFrame({
	    'column': ['Energy', 'Power'],
	    'var_r': [energy_cov_matrix.iloc[0, 0], power_cov_matrix.iloc[0, 0]],
	    'var_i': [energy_cov_matrix.iloc[1, 1], power_cov_matrix.iloc[1, 1]],
	    'cov_ri': [energy_cov_matrix.iloc[0, 1], power_cov_matrix.iloc[0, 1]],
	    'roh': [energy_corrcoef.statistic, power_corrcoef.statistic],
	    'pv': [energy_corrcoef.pvalue, power_corrcoef.pvalue],
	})
	x['rat'] = x.cov_ri / (x.var_i + x.var_r)

	return x


@command('power_snr')
def power_snr(data: GroupedData, label_columns: str, measurement_columns: str, robust=False):
	df = data.df

	label_columns = tuple_list(label_columns)

	reduction_func = 'median' if robust else 'mean'

	results = pd.DataFrame({
	    'count': df.shape[0],
	    'model': ','.join(label_columns),
	}, index=[0])

	for column in tuple_list(measurement_columns):

		df_g = df.groupby(label_columns).agg(  #
		    gmean=pd.NamedAgg(column=column, aggfunc=reduction_func),  #
		    gcount=pd.NamedAgg(column=column, aggfunc='count')  #
		)

		var_noise = df.groupby(label_columns)[column].var(ddof=0).agg(reduction_func)
		var_signal = df_g.gmean.sub(df_g.gmean.mean()).pow(2).mul(df_g.gcount).div(df_g.gcount.sum()).sum()

		snr = var_signal / var_noise

		result = pd.DataFrame(
		    {
		        f'{column}_snr': snr,
		        f'{column}_var_signal': var_signal,
		        f'{column}_var_noise': var_noise,
		    }, index=[0])

		results = pd.concat([results, result], axis=1)

	return results


@command('power_validate')
def power_validate_model(data: GroupedData, stop_if_failed=False):
	ctx_power = data.ctx.module_data['power']

	for model in ctx_power.models:
		public_words = model.public_words()

		class ValidationAdapter:

			def __getitem__(self, key):
				# check if a unknown value
				if key in public_words:
					# convert the public word to a secret word as the HD will use it
					key = key.replace('g', 'v')
					#key = ctx_power.vs_str[ctx_power.gs_str.index(key)]
				return data.df[key]

			def __len__(self):
				return len(data.df)

		for H, mc in zip(model.eval_generator(ValidationAdapter()), model.components):

			if not np.any(H):
				print(f'model "{mc}": {colors.color_fg.green}successfully validated!{colors.color_fg.end}')
			else:
				print(f'model "{mc}": {colors.color_fg.red}VALIDATION FAILED!{colors.color_fg.end}')
				print(H)
				if stop_if_failed:
					breakpoint()

	return data.df


##
# CPA
##


def get_type(m):
	if isinstance(m, rda.power.model.HD):
		return 'hd', str(m.x), str(m.y)
	if isinstance(m, rda.power.model.HW):
		x = get_type(m.x)
		return 'hw' + x[0], x[1], x[2]
	if isinstance(m, rda.power.model.Secret):
		return 'v', str(m.x), ''
	if isinstance(m, rda.power.model.Public):
		return 'g', str(m.x), ''

	return 'unkown', '', ''


##
# Find Model Coefficients
##
@command('power_fit', do_print=False, print_groups=[None, 'model'])
def power_find_model_coefficients(
    data: GroupedData,
    y_column: str,
    independent_coefficients: bool = False,
    all_ones: bool = False,
    window=None,
    scale=1,
    update_model=False,
    use_gpu=True,
):
	ctx = data.ctx
	ctx_power = ctx.module_data['power']

	results = pd.DataFrame()

	# measurements
	measurements = data.df[y_column].to_numpy()

	#@dataclass(frozen=True)
	#@dataclass(frozen=True)
	#class GPUAdapter:
	#	def __getitem__(self, key):
	#		return cp.asarray(data[key].values)

	#if rda.optimized.utils.cupy_available() and use_gpu:
	#	adapter = GPUAdapter()
	#	measurements = cp.asarray(measurements)
	#else:
	#	adapter = data

	# optimized pearson r calculator
	if not independent_coefficients and window:
		pearsonr = rda.optimized.pearsonr.PearsonRCalculatorRolling(measurements, window=window)
	else:
		pearsonr = rda.optimized.pearsonr.PearsonRCalculator(measurements)

	for model in ctx_power.models:
		# strings of components
		comp_strs = list(map(str, model.components))

		if not independent_coefficients:
			# construct the hypothesis matrix
			H = np.vstack(list(model.eval_generator(data.df))).T

			# use a robust regressor
			reg = HuberRegressor(fit_intercept=True).fit(H, measurements)

			# get the r2 score
			r2_score = reg.score(H, measurements)

			# get the coefficients
			coefs = np.array(reg.coef_)

			if all_ones:
				coefs[:] = 1

			# compute the resulting pearson R for the scaling coefficients
			rho, rho_l, rho_u, pv = pearsonr.compute(H.dot(coefs)[np.newaxis, :], confidence=True)

			# create result data frame
			# we only have one hypothesis in this case so the ravel is okay
			# if we use a windowed correlation we need to offset the result as we don't have samples
			result = pd.DataFrame(
			    {
			        'rho': rho.ravel(),
			        'rho_l': rho_l.ravel(),
			        'rho_u': rho_u.ravel(),
			        'pv_rho': pv.ravel(),
			        'r2_score': r2_score,
			        'N': measurements.shape[0],
			        'type': [get_type(m)[0] for m in model.components] if len(model.components) == 1 else None,
			        'x': [get_type(m)[1] for m in model.components] if len(model.components) == 1 else None,
			        'y': [get_type(m)[2] for m in model.components] if len(model.components) == 1 else None,
			    },
			    index=data.df.index[window:] if window is not None else None,
			)
			result[comp_strs] = coefs * scale

		else:

			def compute_parts(H):

				rho, rho_l, rho_u, pv = pearsonr.compute(H[np.newaxis, :], confidence=True)

				reg = HuberRegressor(fit_intercept=True).fit(H.reshape(-1, 1), measurements)

				# get the r2 score
				r2 = reg.score(H.reshape(-1, 1), measurements)

				# get the coefficients
				coefs = np.array(reg.coef_)

				if all_ones:
					coefs[:] = 1

				return rho, rho_l, rho_u, pv, coefs, np.array([r2])

			parts = [
			    compute_parts(H) for H in tqdm(
			        model.eval_generator(data.df),
			        total=len(model),
			        desc='Components',
			        leave=False,
			    )
			]

			rho, rho_l, rho_u, pv, coefs, r2_score = np.concatenate(parts, axis=1)

			# create result data frame
			result = pd.DataFrame({
			    'comp': comp_strs,
			    'rho': rho,
			    'rho_l': rho_l,
			    'rho_u': rho_u,
			    'pv_rho': pv,
			    'r2_score': r2_score,
			    'N': measurements.shape[0],
			    'coef': coefs * scale,
			    'type': [get_type(m)[0] for m in model.components],
			    'x': [get_type(m)[1] for m in model.components],
			    'y': [get_type(m)[2] for m in model.components],
			})

		if update_model:
			model.set_coefficients(coefs)
			print(f'{model!r}')

		result['model'] = str(model)

		results = pd.concat([results, result], ignore_index=False)

	return results


##
# CPA
##
@command('power_cpa')  # print=True, print_groups=['model']
def power_cpa(
    data: GroupedData,
    column: str,
    constant_secret_values=True,
    normalization='POST',
    use_gpu: bool = True,
    step: int = 0,
    blocked: bool = False,
):
	df = data.df
	ctx = data.ctx
	rda.utils.utils.valid_columns(df, [column])
	ctx.print_group_header(df)
	ctx_power = ctx.module_data['power']

	# create result data frame format
	results = pd.DataFrame()

	# if we have an iterative CPA set the size range - if we don't set it to the data frame size
	sizes = range(step, df.index.shape[0], step) if step != 0 else [df.index.shape[0]]

	for size in sizes:

		for (model) in ctx_power.models:
			# get the secrets we want to recover
			secret_words = model.secret_words()

			# if we want to have constant secret values we need to group by all secret data
			grouping_secret_words = (ctx_power.vs_str if constant_secret_values else secret_words)

			# ignore the pandas warning for groupby tuples and group the data
			with catch_warnings():
				# TODO: fix this
				simplefilter(action='ignore', category=FutureWarning)
				value_groups = df.groupby(grouping_secret_words)

			# perform the cpa for each constant victim value in the df
			for ground_truth, x in value_groups:

				# check if we want ot have an iterative CPA, a blocked one, or one over all samples
				if size != 0 and blocked:
					samples = x[size - step:size]

				elif size != 0:
					samples = x[:size]

				else:
					samples = x

				r = power_cpa_one_value(
				    df=samples,
				    ctx_power=ctx_power,
				    column=column,
				    model=model,
				    normalization=normalization,
				    use_gpu=use_gpu,
				)

				r = r[['rank', 'N', 'model', 'est']].iloc[0:1]

				if size != 0 and blocked:
					r['N'] = size

				results = pd.concat([results, r], ignore_index=False)

	return df if step == 0 else results


def power_cpa_one_value(df: pd.DataFrame, ctx_power, column: str, model, normalization, use_gpu):
	to_ascii = False
	to_binary = True

	# perform the cpa
	if rda.optimized.utils.cupy_available() and use_gpu:
		results = rda.power.cpa.compute_cpa_gpu(df, ctx_power, column, model, normalization, only_estimate_rank=False)
	else:
		results = rda.power.cpa.compute_cpa(df, ctx_power, column, model, normalization)

	# get the model's secret words
	secret_words = model.secret_words()

	# and the ground truth which is constant over the whole df (grouped outside)
	ground_truth = df[secret_words].iloc[0].values

	results['model'] = str(model)

	rank = results['rank'].iloc[0]
	rho_rank = results['rho_gt'].iloc[0]
	est_char = '_est' if results['est'].iloc[0] else ''

	results['N'] = df.index.shape[0]

	results['cor'] = np.all(results[secret_words].values == ground_truth, axis=1)

	fmt = [
	    format_cpa_result(
	        ground_truth,
	        results[secret_words].iloc[i].values,
	        results.rho.iloc[i],
	        to_ascii=to_ascii,
	        to_binary=to_binary,
	    ) for i in range(min(2, results.shape[0]))
	]

	S = format_value(np.array(secret_words))
	V = format_value(ground_truth, to_ascii=to_ascii, to_binary=to_binary)
	# .3g
	print(
	    f'{str(model):30} N={df.index.shape[0]:} S={S} V={V} {{R{est_char}={rank:6} r={rho_rank:.3e}}} {" ".join(fmt)}')

	#results = results.rename(columns={old: new for old, new in zip(secret_words, val)})

	return results


def format_value(value, to_ascii=False, to_binary=False):

	def cvt(x):
		return chr(x) if chr(x).isprintable() else ' '

	if to_ascii:
		return ','.join([f"'{cvt(x):1}'" for x in value])
	if to_binary:
		return np.array2string(value, formatter={'all': '{:08b}'.format})
	else:
		return np.array2string(value, formatter={'all': '{:3}'.format})


def format_cpa_result(ground_truth, candidate, rho, to_ascii=False, to_binary=False):
	lbl = colors.color_fg.lightblue
	lcy = colors.color_fg.lightcyan
	gre = colors.color_fg.green
	ora = colors.color_fg.orange
	end = colors.color_fg.end

	un = '\033[4m'
	eu = '\033[24m'

	gt_str = format_value(ground_truth, to_ascii=to_ascii, to_binary=to_binary)
	candidate_str = format_value(candidate, to_ascii=to_ascii, to_binary=to_binary)
	distance_str = format_value(rda.optimized.functions.hd(candidate, ground_truth))

	final_str = ''.join(
	    [f'{un}{char}{eu}' if char != gt_char else f'{char}' for char, gt_char in zip(candidate_str, gt_str)])

	return f'{lbl}{final_str} {gre}Δ={distance_str} {lcy}{rho:.3e}  {ora}|{end}'
