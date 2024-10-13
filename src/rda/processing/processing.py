import rda.utils.utils
from functools import wraps
import pandas as pd
import numpy as np
from rda.cli.command import GroupedData, UngroupedData, command, tuple_list


def report_dropped_samples(function):

	@wraps(function)
	def wrapper(data, *args, **kwargs):
		df = data.df
		n = df.index.shape[0]
		df = function(data, *args, **kwargs)
		n_removed = n - df.index.shape[0]
		per = n_removed / n * 100 if n != 0 else 100
		print(f"removed {n_removed:10d} samples ({per:5.1f}%)")
		return df

	return wrapper


def foreach_column_masked(name):

	def decorator(function):

		@wraps(function)
		def wrapper(data, *args, **kwargs):
			df = data.df

			def ww(*args, **kwargs):
				return rda.utils.utils.get_named_argument(function, name, args, kwargs)

			columns = tuple_list(ww(df, *args, **kwargs))
			rda.utils.utils.valid_columns(df, columns)
			mask = None
			for c in columns:
				lower, upper = function(df[c], *args, **kwargs)
				if np.all(lower < upper):
					m = (df[c] >= lower) & (df[c] <= upper)
				else:
					m = (df[c] >= lower) | (df[c] <= upper)

				if mask is not None:
					mask &= m
				else:
					mask = m

			return df[mask]

		return wrapper

	return decorator


def run_code(df, code):
	if df.index.shape[0] == 0:
		return df

	#print(f'- Running {code!r} ')  # , end=''

	local_vars = {'x': df, 'np': np}

	while True:
		try:
			exec(code, local_vars)
			break
		except NameError as e:
			var = str(e).split('\'')[1]  # todo fix with python 3.10
			print(f'- converting variable {var} to string ')  # , end=''
			local_vars[var] = var
		except Exception as e:
			print(f'\ncode {code!r} raised exception: {e}')
			exit(-1)

	# for c in df.columns:
	#    if pd.api.types.is_categorical_dtype(df[c]):
	#        df[c] = df[c].cat.remove_unused_categories()

	return local_vars['x']


@command('filter')
def moving_filter(data: GroupedData, columns, filter_type: str = 'mean', number_samples: int = 100):
	columns = tuple_list(columns)

	rda.utils.utils.valid_columns(data.df, columns)

	for c in columns:
		if filter_type == 'mean':
			data.df[c] -= data.df[c].rolling(number_samples, center=True, min_periods=1).mean()
		elif filter_type == 'median':
			data.df[c] -= data.df[c].rolling(number_samples, center=True, min_periods=1).median()
		else:
			print(f'unknown filter: {filter_type}')
			exit(-1)
	return data.df


@command('rolling')
def rolling(data: GroupedData, columns, function, number_samples: int):
	columns = tuple_list(columns)
	rda.utils.utils.valid_columns(data.df, columns)
	for c in columns:
		data.df[c] = data.df[c].rolling(number_samples, center=False, min_periods=number_samples).aggregate(function)
	return data.df


@command('sel')
@report_dropped_samples
def select(data: GroupedData, code: str):
	return run_code(data.df, f'x=x[{code}]')


@command('e')
def execute(data: GroupedData, code: str):
	return run_code(data.df, code)


@command('per')
@report_dropped_samples
@foreach_column_masked('outliers')
def remove_outliers_percentile(x, outliers, lower: float, upper: float):
	l = x.quantile(lower / 100)
	u = x.quantile(upper / 100)
	return l, u


@command('std')
@report_dropped_samples
@foreach_column_masked('outliers')
def remove_outliers_standard(x, outliers, lower: float, upper: float):
	l = x.mean() + lower * x.std()
	u = x.mean() + upper * x.std()
	return l, u


@command('cut')
@report_dropped_samples
@foreach_column_masked('outliers')
def remove_outliers_cut(x, outliers, lower: float, upper: float):
	return lower, upper


@command('mvper')
@report_dropped_samples
@foreach_column_masked('outliers')
def remove_outliers_moving_percentile(x, outliers, lower: float, upper: float, N: int):
	u = x.rolling(N, center=True, min_periods=1).quantile(upper / 100)
	l = x.rolling(N, center=True, min_periods=1).quantile(lower / 100)
	return l, u


@command('mvstd')
@report_dropped_samples
@foreach_column_masked('outliers')
def remove_outliers_moving_std(x, outliers, lower: float, upper: float, N: int):
	s = x.rolling(N, center=True, min_periods=1).std()
	m = x.rolling(N, center=True, min_periods=1).mean()
	l = m + lower * s
	u = m + upper * s
	return l, u


@command('norm')
def norm(data: GroupedData, columns, N=None, norm_std=True):

	columns = tuple_list(columns)

	if N is None:
		m = data.df[columns].mean()
		s = data.df[columns].std()
	else:
		m = data.df[columns].rolling(window=N, min_periods=N).mean()
		s = data.df[columns].rolling(window=N, min_periods=N).std()

	data.df[columns] = (data.df[columns] - m)

	if norm_std:
		data.df[columns] = data.df[columns] / s

	return data.df


@command('norm_minmax')
def norm(data: GroupedData, columns, N=None):

	columns = tuple_list(columns)

	if N is None:
		col_min = data.df[columns].min()
		col_max = data.df[columns].max()
	else:
		col_min = data.df[columns].rolling(window=N, min_periods=N).min()
		col_max = data.df[columns].rolling(window=N, min_periods=N).max()

	data.df[columns] = (data.df[columns] - col_min) / (col_max - col_min)

	return data.df


@command('dropna')
@report_dropped_samples
def dropna(data: GroupedData, columns: tuple = None):
	with pd.option_context('mode.use_inf_as_na', True):
		return data.df.dropna(axis=0, subset=columns)


@command('dropna_cols')
@report_dropped_samples
def dropna(data: GroupedData, columns: tuple = None):
	with pd.option_context('mode.use_inf_as_na', True):
		return data.df.dropna(axis=1, subset=columns)


from scipy.stats import iqr

lookup = {
    'iqr': iqr,
    'q20': lambda x: x.quantile(0.2),
    'q80': lambda x: x.quantile(0.8),
}


@command('reduce')
def reduce(data: GroupedData, columns, function: str):
	columns = tuple_list(columns)
	x = data.df[columns].aggregate(lookup[function] if function in lookup else function)
	return pd.DataFrame({c: x[c] for c in columns}, index=[0])


@command('reduce_g')
def reduce_g(data: GroupedData, groups, columns, function: str, keep_first: bool = False):
	groups = tuple_list(groups)
	columns = tuple_list(columns)

	aggregates = {c: lookup[function] if function in lookup else function for c in columns}

	if keep_first:
		aggregates.update({c: 'first' for c in data.df.columns if c not in columns})

	#print(aggregates)

	#breakpoint()
	return data.df.groupby(groups).agg(aggregates)
	return pd.DataFrame({c: x[c] for c in columns})


@command('transform_g')
def transform_g(data: GroupedData, groups, columns, function: str):
	groups = tuple_list(groups)
	columns = tuple_list(columns)

	function = lookup[function] if function in lookup else function

	data.df.loc[:, columns] = data.df.groupby(groups)[columns].transform(function)

	return data.df


@command('add_upper_lower')
def add_upper_lower(data: GroupedData, column: str, column_l: str, column_u: str):
	df = pd.concat([data.df, data.df.rename({column_l: column}),
	                data.df.rename({column_u: column})],
	               ignore_index=False)
	print(df)
	return df


@command('get_stats')
def get_stats(
    data: GroupedData,
    columns,  #: str | tuple[str],
    number_samples=100,
):
	columns = tuple_list(columns)
	rda.utils.utils.valid_columns(data.df, columns)

	df = pd.DataFrame(index=data.df.index)

	for c in columns:
		r = data.df[c].rolling(number_samples, center=False, min_periods=number_samples)

		df[f'{c}_mean'] = r.mean()
		df[f'{c}_std'] = r.std()
		df[f'{c}_median'] = r.median()
		df[f'{c}_p95'] = r.quantile(0.95)
		df[f'{c}_p05'] = r.quantile(0.05)
		df[f'{c}_min'] = r.min()
		df[f'{c}_max'] = r.max()
		df[f'{c}_range'] = df[f'{c}_p95'] - df[f'{c}_p05']

	return df.dropna()


@command('to_time')
def to_time(data: GroupedData, column: str):
	data.df[column] = pd.to_datetime(data.df[column], unit='s')  #.tz_convert('Europe/Berlin')
	return data.df
