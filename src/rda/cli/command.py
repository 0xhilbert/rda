import inspect

import pandas as pd

from dataclasses import dataclass, field
from functools import wraps
from typing import List, Union, Any, Dict

import rda.utils.utils
import rda.utils.colors as colors

cor = colors.color_fg.orange
red = colors.color_fg.red
cgr = colors.color_fg.green
ccy = colors.color_fg.cyan
clc = colors.color_fg.lightcyan
ce = colors.color_fg.end


@dataclass(frozen=True)
class Context:
	''' class to store metadata information which is passed along side with the data frame to each command'''

	# the current group keys
	groups: List[str] = field(default_factory=lambda: ['_'])

	# if the current data frame is saved the groups and data are stored here
	saved_groups: Dict[str, List[str]] = field(default_factory=lambda: {})
	saved_dfs: Dict[str, List[pd.DataFrame]] = field(default_factory=lambda: {})

	# structure to store metadata for the extensions
	module_data: Dict[str, Any] = field(default_factory=lambda: {})

	def save(self, dfs: List[pd.DataFrame], id: str) -> None:
		'''save the dfs and the grouping information'''
		self.saved_groups[id] = self.groups.copy()
		self.saved_dfs[id] = dfs

	def load(self, id: str) -> List[pd.DataFrame]:
		'''load the dfs and the grouping information'''
		self.groups.clear()
		self.groups.extend(self.saved_groups[id])
		return self.raw_load(id)

	def raw_load(self, id: str) -> List[pd.DataFrame]:
		'''access the saved dfs directly'''
		return self.saved_dfs[id]

	def get_group_id(self, df: pd.DataFrame):
		'''return the group identifier'''
		if self.groups == ['_']:
			return (1,)
		return tuple(df[self.groups].iloc[0],)

	def is_grouped(self) -> bool:
		'''check if the data is currently grouped'''
		return self.groups != ['_']

	@staticmethod
	def format_dict(x: Dict, key_color: str = '', value_color: str = '') -> str:
		''' format a dict with nice colors '''

		return ' '.join(
		    [f'{key_color}{k}{colors.color_fg.end}={value_color}{v!r}{colors.color_fg.end}' for k, v in x.items()])

	def print_header(self, function, args, kwargs, newline=True):

		spec = inspect.getfullargspec(function)

		args_str = Context.format_dict(dict(zip(spec.args[1:], args)), key_color=clc)
		kwargs_str = Context.format_dict(kwargs, key_color=clc)

		print(
		    f'{ccy}>{ce} {cor}{function.__name__}{ce} {args_str} {kwargs_str}\t',
		    flush=True,
		    end=None if newline else ' ')

	def print_group_header(self, df, newline=True):
		if df is None or not self.is_grouped():
			return
		v = Context.format_dict(dict(zip(self.groups, self.get_group_id(df))), key_color=cor)
		print(f'{ccy}>>{ce} {v}\t', flush=True, end=None if newline else ' ')

	def do_grouping(self, dfs, groups):
		from warnings import catch_warnings, simplefilter

		if self.is_grouped():
			dfs = self.undo_grouping(dfs)

		df = dfs[0]

		rda.utils.utils.valid_columns(df, groups)

		with catch_warnings():
			simplefilter(action='ignore', category=FutureWarning)
			if df.index.shape[0] == 0:
				print('error empty data frame cannot be grouped')
				exit(0)

			_, value_dfs = zip(*df.groupby(groups, group_keys=True, as_index=False))

		self.groups.clear()
		self.groups.extend(groups)

		return list(value_dfs)

	def undo_grouping(self, dfs):
		self.groups.clear()
		return [pd.concat(dfs)]

	def do_merging(self, id_column, ids):

		def load_dfs(id):
			df = pd.concat(self.raw_load(id))  # ungroup just to make sure
			df[id_column] = id
			return df

		dfs = [load_dfs(id) for id in ids]

		return self.undo_grouping(dfs)

	def add_group_ids(self, dfs, ids):

		def agi(df, id):
			df.loc[:, self.groups] = id
			return df

		assert (len(dfs) == len(ids))
		return [agi(df, id) for df, id in zip(dfs, ids)]


def tuple_list(x):
	''' functrion to make it easier to interact with fire variadic arguments '''
	if isinstance(x, tuple):
		return list(x)
	if isinstance(x, list):
		return x
	return [x]


@dataclass
class RapidDataAnalysis:
	''' root structure - we inject each command into this class'''

	dfs: List[pd.DataFrame] = field(default_factory=lambda: [])
	ctx: Context = Context()

	def __str__(self):
		return ""

	def copy(self):
		return RapidDataAnalysis(self.dfs.copy(), self.ctx)


@dataclass
class GroupedData:
	''' grouped data represents a single data frame potential associated with a group '''
	df: pd.DataFrame
	ctx: Context


@dataclass
class UngroupedData:
	''' ungrouped data represents all data frames of each group '''

	dfs: List[pd.DataFrame]
	ctx: Context

	def copy(self):
		return RapidDataAnalysis(self.dfs, self.ctx)


from contextlib import contextmanager


@contextmanager
def add_groups(data: RapidDataAnalysis, *groups):
	''' helper to add additional group within recusive commands '''

	existing_groups = data.ctx.groups.copy()
	data.g((*existing_groups, *groups))
	try:
		yield data
	finally:
		if len(existing_groups):
			data.g(tuple(existing_groups))
		else:
			data.u()


def remove_empty_dfs(dfs):
	return [x for x in dfs if x.index.shape[0] != 0]


def compute_total_length(dfs):
	try:
		return sum([x.shape[0] if x is not None else 0 for x in dfs])
	except:
		breakpoint()


def command(name, grouped=True, newline=True, do_print=False, print_groups=[]):
	''' decorator to define a command '''

	def decorator(command):

		@wraps(command)
		def dispatcher(data: RapidDataAnalysis, *args, **kwargs):

			data.ctx.print_header(command, args, kwargs, newline=data.ctx.is_grouped() or newline)

			data.dfs = remove_empty_dfs(data.dfs)
			if len(data.dfs) == 0:
				print('all data frames are empty!')

			group_ids = [data.ctx.get_group_id(df) for df in data.dfs]
			groups = data.ctx.groups.copy()

			len_before = compute_total_length(data.dfs)

			if grouped:
				returned_dfs = [command(GroupedData(df, data.ctx), *args, **kwargs) for df in data.dfs]
				returned_dfs = [x if x is not None else df for df, x in zip(data.dfs, returned_dfs)]
			else:
				returned_dfs = command(UngroupedData(data.dfs, data.ctx), *args, **kwargs)

			if returned_dfs is not None:
				data.dfs = returned_dfs

			len_after = compute_total_length(data.dfs)

			# if the groups did not change add the group ids
			# otherwise we executed a group or ungroup command
			if groups == data.ctx.groups:
				data.dfs = data.ctx.add_group_ids(data.dfs, group_ids)

			if do_print:
				groups = sum([[x] if x is not None else data.ctx.groups for x in print_groups], [])
				x = pd.concat(data.dfs).set_index(groups).sort_index()
				rda.utils.utils.print_all(x)

			if len_before != len_after:
				print(f'size changed from {len_before} to {len_after} samples')

			return data

		# inject the generated command to be accessible via the fire cli
		setattr(RapidDataAnalysis, name, dispatcher)
		return dispatcher

	return decorator
