import cupy as cp
import numpy as np
import pandas as pd
import textwrap

def split_comma(ctx, params, value):
	return value.split(',') if value else []

def unique_ordered(l):
	visited = set()
	return [x for x in l if not (x in visited or visited.add(x))]

def valid_columns(df, columns):
	for c in columns:
		if c not in df:
			print()
			print(f'unknown column: {c}')
			cols = ','.join(map(repr, df.columns))
			print(f'available columns: {cols}')
			exit(-1)
	return True

def print_all(df):
	with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
		print(df)

def get_named_argument(function, name, args, kwargs):
	import inspect
	index = inspect.getfullargspec(function).args.index(name)
	if index < len(args):
		return args[index]
	else:
		return kwargs[name]


def groupby(df, groups):
	if groups:
		return df.groupby(groups, group_keys=False, as_index=False)
	else:
		return df.groupby(['_'], group_keys=False, as_index=False)
