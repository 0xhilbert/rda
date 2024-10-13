from rda.cli.command import GroupedData, UngroupedData, command, tuple_list
from typing import List
from rda.utils.utils import print_all

import pandas as pd
import numpy as np

##
# Loading and Saving Data
##


def read_cvs_file(file_name, number_of_rows=None):
	# we need to manually convert value and guess bytes to byte arrays
	converters = {
	    'Value': lambda x: bytes(reversed(bytes.fromhex(x[2:]))),
	    'Guess': lambda x: bytes(reversed(bytes.fromhex(x[2:]))),
	}
	return pd.read_csv(file_name, comment='#', nrows=number_of_rows, converters=converters)


def read_npy_file(file_name, number_of_rows=None):
	# load the npy file and convert it to a rec array
	# extend the header size so we can extract the metadata
	data = np.load(file_name, max_header_size=10000000000).view(np.recarray)

	columns = list(data.dtype.names)

	# check if we have a metadata column
	if 'metadata' in columns:
		# if so the metadata is the 'title' of the metadata columns
		metadata = data.dtype.fields['metadata'][2]
		# remove the meta data column as otherwise printing is ugly
		columns.remove('metadata')
		data = data[columns]

	# only use up to number_of_rows rows
	if number_of_rows is not None:
		data = data[:number_of_rows]

	df = pd.DataFrame(data, copy=False)  #.convert_dtypes(dtype_backend='pyarrow')

	# todo fix types here
	return df


def read_feather_file(file_name, number_of_rows=None):
	df = pd.read_feather(file_name)

	# only use up to number_of_rows rows
	if number_of_rows is not None:
		df = df.iloc[:number_of_rows]

	return df


def read_parquet_file(file_name, number_of_rows=None):
	df = pd.read_parquet(file_name)

	# only use up to number_of_rows rows
	if number_of_rows is not None:
		df = df.iloc[:number_of_rows]

	return df


def read_file(file_name, nrows, add_file_info):

	if file_name.endswith('.csv'):
		df = read_cvs_file(file_name, nrows)

	elif file_name.endswith('.npy'):
		df = read_npy_file(file_name, nrows)

	elif file_name.endswith('.feather'):
		df = read_feather_file(file_name, nrows)

	elif file_name.endswith('.parquet'):
		df = read_parquet_file(file_name, nrows)

	else:
		raise RuntimeError(f'no import functions for: {file_name}')

	if add_file_info:
		df['File'] = file_name

	print(f'- read file {file_name!r} with {df.index.shape[0]} rows - columns: {", ".join(df.columns)}')
	return df


@command('cvt_npy_to_feather', grouped=False)
def convert_npy_to_feather(data: UngroupedData, file_name: str, output_name):
	'''helper function if npy files get to large'''
	data = np.load(file_name, max_header_size=10000000000, mmap_mode='r').view(np.recarray)

	columns = list(data.dtype.names)

	# check if we have a metadata column
	if 'metadata' in columns:
		# if so the metadata is the 'title' of the metadata columns
		metadata = data.dtype.fields['metadata'][2]
		# remove the meta data column as otherwise printing is ugly
		columns.remove('metadata')
		data = data[columns]

	import pyarrow as pa
	import pyarrow.feather as feather

	schema = pa.schema([
	    ('Guess', pa.binary(data.dtype['Guess'].itemsize)),
	    ('Value', pa.binary(data.dtype['Value'].itemsize)),
	])

	test_table = pa.Table.from_pydict({k: data[k][0:1] for k in data.dtype.fields.keys() if k not in ['Guess', 'Value']})

	schema = pa.unify_schemas([test_table.schema, schema])

	table = pa.Table.from_pydict({k: data[k] for k in data.dtype.fields.keys()}, schema=schema)

	feather.write_feather(table, output_name, compression='uncompressed')
	return data.dfs


@command('i', grouped=False)
def open_files(data: UngroupedData, file_names, nrows: int = None, add_file_info: bool = False) -> List[pd.DataFrame]:
	dfs = [read_file(f, nrows, add_file_info) for f in tuple_list(file_names)]

	df = pd.concat(dfs)

	if add_file_info and 'original_file_index' not in df:
		df = df.reset_index(names=['original_file_index'])

	return data.ctx.undo_grouping([df])


@command('cache_store', grouped=False)
def cmd_store(data: UngroupedData, file_name: str) -> List[pd.DataFrame]:
	pd.concat(data.dfs).to_feather(file_name)
	return data.dfs


@command('cache_load', grouped=False)
def cmd_store(data: UngroupedData, file_name: str) -> List[pd.DataFrame]:
	df = pd.read_feather(file_name)
	return data.ctx.undo_grouping([df])


@command('drop')
def cmd_drop(data: GroupedData, column_names) -> List[pd.DataFrame]:
	return data.df.drop(tuple_list(column_names), axis=1)


@command('exit')
def cmd_exit(data: GroupedData):
	exit(0)


@command('store_csv', grouped=False)
def store_csv_file(data: UngroupedData, file_name: str, columns=None) -> List[pd.DataFrame]:
	'''store csv file'''
	x = pd.concat(data.dfs)
	if columns is None:
		columns = x.columns
	else:
		columns = tuple_list(columns)

	print(f'string columns {columns}')
	x[columns].to_csv(file_name)
	return data.dfs


@command('meta', grouped=False)
def extract_meta(data: UngroupedData, file_name: str):
	'''extract the meta data form a data file'''

	from tempfile import TemporaryDirectory
	from os.path import basename
	from subprocess import run

	if file_name.endswith('.csv'):
		lines = open(file_name, 'r').readlines()
		metadata = [line[1:] for line in lines if line.startswith('#')]

	elif file_name.endswith('.npy'):
		data = np.load(file_name, max_header_size=10000000000).view(np.recarray)
		metadata = data.dtype.fields['metadata'][2]

	else:
		raise NotImplementedError(f'metadata for file type of {file_name} not supported')

	with TemporaryDirectory(prefix=f'{basename(file_name)}') as tmp_directory:

		with open(f'{tmp_directory}/invoke.sh', 'w') as invoke_file:
			invoke_file.writelines(metadata)

		# dangerous
		input('press enter to run meta data script - this could be dangerous if the file is untrusted!')

		run('chmod +x invoke.sh; ./invoke.sh', shell=True, text=True, cwd=tmp_directory)
		run(f'code -n {tmp_directory}', shell=True, text=True)

		print(f'directory: {tmp_directory}')

		input('Press any key to continue')


##
# Grouping and un-grouping of data
##


@command('g', grouped=False)
def do_grouping(data: UngroupedData, columns) -> List[pd.DataFrame]:
	return data.ctx.do_grouping(data.dfs, tuple_list(columns))


@command('u', grouped=False)
def undo_grouping(data: UngroupedData) -> List[pd.DataFrame]:
	return data.ctx.undo_grouping(data.dfs)


##
# Intermediate Results
##


@command('save', grouped=False)
def save_data(data: UngroupedData, id: str) -> List[pd.DataFrame]:
	'''save intermediate data'''
	data.ctx.save(data.dfs, id)
	return data.dfs


@command('load', grouped=False)
def load_data(data: UngroupedData, id: str) -> List[pd.DataFrame]:
	'''load intermediate data'''
	return data.ctx.load(id)


@command('merge', grouped=False)
def merge_data(data: UngroupedData, id_column: str, ids) -> List[pd.DataFrame]:
	'''merge multiple saved data frames'''
	return data.ctx.do_merging(id_column, tuple_list(ids))


@command('print', grouped=False)
def print_data(data: UngroupedData, n: int = None, index=None) -> List[pd.DataFrame]:
	'''print current data frame'''
	index = tuple_list(index) if index else data.ctx.groups

	x = pd.concat(data.dfs)

	if len(index):
		x = x.set_index(index)
		x = x.sort_index()

	n = n or x.index.shape[0]
	print_all(x.head(n))
	return data.dfs


@command('desc')
def describe_data(data: GroupedData, columns=None) -> List[pd.DataFrame]:
	'''describe current data frame'''
	x = data.df

	if columns:
		x = x[tuple_list(columns)]

	print(x.describe())
	return data.df


@command('take')
def take_data(data: GroupedData, n: int) -> pd.DataFrame:
	'''take all elements where: x.index < n'''
	return data.df.head(n)


@command('take_g')
def take_subgroup(data: GroupedData, columns) -> pd.DataFrame:
	return next(iter(data.df.groupby(tuple_list(columns))))[1]


@command('dbg', grouped=False)
def dbg(data: UngroupedData) -> List[pd.DataFrame]:
	'''debug breakpoint to inspect data'''
	df = data.dfs[0] if len(data.dfs) else []
	breakpoint()
	return data.dfs


@command('blocks')
def blocks(data: GroupedData, block_size) -> List[pd.DataFrame]:
	'''split the index into discrete blocks'''

	if isinstance(block_size, str):
		# if we get passed a string we assume that we have a time series
		new_index = data.df.index.round(block_size)

	elif isinstance(block_size, int):
		# if we get passed an int we assume that we have discrete blocks
		new_index = np.repeat(data.df.index[::block_size], block_size)[:data.df.index.shape[0]]

	data.df['blocks'] = new_index
	data.df.set_index(new_index, inplace=True, drop=True)
	return data.df


##
# Processing
##


@command('idx')
def index(data: GroupedData, columns):
	'''set data frame index'''
	return data.df.set_index(tuple_list(columns))


@command('decode_str')
def decode_str(data: GroupedData, column: str):
	data.df.loc[:, column] = data.df[column].str.decode(encoding='ascii')
	return data.df


@command('rec_dur')
def recording_duration(data: GroupedData, column: str):
	dur = pd.to_datetime(data.df[column].iloc[[0, -1]], unit='s').diff()
	print(f'The data was recorded over {dur}')
	return data.df
