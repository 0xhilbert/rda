#!/usr/bin/env python3
import rda.utils
import scipy

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import seaborn as sns

from tqdm import tqdm

from dataclasses import dataclass, field

import functools

from rda.cli.command import GroupedData, UngroupedData, command, tuple_list

LABELSIZE = 6

from rda.plot.scrollable_legend import add_scrollable_legend


def to_label(xs):
	return ' '.join([str(x) for x in xs])


def to_str(df, groups):
	return df[groups].apply(lambda x: to_label(x), axis=1)


#@functools.cache
def to_hue(df, groups):
	assert (groups)

	# only first group gets color
	if len(groups) > 1:
		groups = groups[0:1]

	if len(groups) == 1:
		y = df[groups[0]]
	else:
		y = df[groups].to_records(index=False).tolist()

	hue = pd.Series(data=y, index=df.index)
	hue.name = ', '.join(groups)
	return hue


#@functools.cache
def to_hue2(df, groups):
	if len(groups) == 0:
		return df, {}

	# only first group gets color
	if len(groups) > 1:
		groups = groups[0:]

	if len(groups) == 1:
		y = df[groups[0]]
	else:
		y = df[groups].to_records(index=False).tolist()

	hue = pd.Series(data=y, index=df.index)
	hue.name = ', '.join(groups)

	df[hue.name] = hue

	return df, {'hue': hue.name, 'hue_order': hue.unique()}


#@functools.cache
def to_style(df, groups):
	assert (groups)

	# solve other groups with styles
	if len(groups) > 1:
		groups = groups[1:]

	if len(groups) == 1:
		y = df[groups[0]]
	else:
		y = df[groups].to_records(index=False).tolist()

	style = pd.Series(data=y, index=df.index)
	style.name = ', '.join(groups)
	return style


#@functools.cache
def to_style2(df, groups):
	if len(groups) == 0:
		return df, {}

	# solve other groups with styles
	if len(groups) > 1:
		groups = groups[1:]

	if len(groups) == 1:
		y = df[groups[0]]
	else:
		y = df[groups].to_records(index=False).tolist()

	style = pd.Series(data=y, index=df.index)
	style.name = ', '.join(groups)

	df[style.name] = style

	return df, {'style': style.name, 'style_order': style.unique()}


###
# PROCESSING
###


def calculate_similarity_matricies(df, groups, column):
	rda.utils.valid_columns(df, [column])

	d = rda.utils.groupby(df, groups)[column]

	N = len(d)

	KS = np.zeros((N, N)) * np.nan
	MW = np.zeros((N, N)) * np.nan

	labels = []

	for i, (name, x) in tqdm(enumerate(d), desc='calculate_similarity_matricies...', leave=False, total=N):
		if len(groups) == 1:
			name = [name]

		labels.append(to_label(name))

		for j, (_, y) in tqdm(enumerate(d), desc='calculate_similarity_matricies...', leave=False, position=1, total=N):
			if i >= j:
				continue

			res = 1 - scipy.stats.kstest(x, y).pvalue
			KS[i, j] = res
			KS[j, i] = res

			res = 1 - scipy.stats.mannwhitneyu(x, y).pvalue
			MW[i, j] = res
			MW[j, i] = res
	return (pd.DataFrame(data=KS, index=labels, columns=labels), pd.DataFrame(data=MW, index=labels, columns=labels))


###
# LEGEND
###

###
# PLOTTING
###


def create_new_axis(window_title, rows=1, cols=1, sharex=False, sharey=False, legacy=True):
	fig, ax = plt.subplots(rows, cols, sharex=sharex, sharey=sharey, squeeze=False)
	fig.canvas.manager.set_window_title(window_title)

	if cols == 1:
		ax = [ax]

	if rows == 1 and not legacy:
		return [ax]

	# tmp.set_title(window_title)

	return ax


@command('plot_filterd', grouped=False)
def plot_samples_histogram(
        data: UngroupedData,
        columns,  #str | tuple[str]
        split_groups: bool = False,
        split_columns: bool = False,
        in_ax=None,
        interpolate: bool = False):

	for df, group_name, column_name, hue, style, palette, ax in iterate_plotting_info(
	    data, columns, split_groups, split_columns, in_ax):

		N = 100

		if interpolate:
			lvl = f'level_{len(meta.groups)}'
			y = df.pivot(columns=meta.groups, values=column)
			scales = y.index.shape[0] / y.count()
			y = y.interpolate(axis=0)
			for c, scale in zip(y.columns, scales):
				y[c] = y[c].rolling(int(N * scale), center=True, min_periods=1).mean()
			y = y.unstack().reset_index(name=column)
			x = lvl
		else:
			y = rda.utils.groupby(df, meta.groups)[column].rolling(N, center=True, min_periods=1).mean().reset_index()
			x = 'index'

		if y[column].isnull().all():
			print('Cannot use moving average ... to few samples!')
			return

		# we need to recompute the hue and style
		hue = to_hue(y, meta.groups)
		style = to_style(y, meta.groups)

		# plot the filtered values
		ax = in_ax or create_new_axis(f'Filtered plot of {column!r}')
		ax = sns.lineplot(
		    data=y,
		    x=x,
		    y=column,
		    hue=hue,
		    hue_order=plot_meta.hue_order,
		    style=style,
		    style_order=plot_meta.style_order,
		    ax=ax,
		    palette=plot_meta.palette,
		    legend=True)

		ax.set_title(f'Filtered Samples of {column!r} {"w" if interpolate else "wo"} interp.')
		ax.set_xlabel('index')
		ax.grid(True)

		if not in_ax:
			add_scrollable_legend(ax)

		ax.set_title(f'Histogram plot of {column_name!r} for {group_name!r}')
		ax = sns.histplot(
		    data=df,
		    x=column_name,
		    **hue,
		    **palette,
		    ax=ax,
		    bins=bins,
		    legend=True,
		    element='bars',
		    stat='count',
		    fill=True,
		    kde=False)
		ax.grid(True)

		if not in_ax:
			add_scrollable_legend(ax)


def plot_samples_filtered(df, meta, column, in_ax=None, interpolate=True):
	rda.utils.valid_columns(df, [column])
	plot_meta = get_plot_meta(df, meta)

	N = 100

	if interpolate:
		lvl = f'level_{len(meta.groups)}'
		y = df.pivot(columns=meta.groups, values=column)
		scales = y.index.shape[0] / y.count()
		y = y.interpolate(axis=0)
		for c, scale in zip(y.columns, scales):
			y[c] = y[c].rolling(int(N * scale), center=True, min_periods=1).mean()
		y = y.unstack().reset_index(name=column)
		x = lvl
	else:
		y = rda.utils.groupby(df, meta.groups)[column].rolling(N, center=True, min_periods=1).mean().reset_index()
		x = 'index'

	if y[column].isnull().all():
		print('Cannot use moving average ... to few samples!')
		return

	# we need to recompute the hue and style
	hue = to_hue(y, meta.groups)
	style = to_style(y, meta.groups)

	# plot the filtered values
	ax = in_ax or create_new_axis(f'Filtered plot of {column!r}')
	ax = sns.lineplot(
	    data=y,
	    x=x,
	    y=column,
	    hue=hue,
	    hue_order=plot_meta.hue_order,
	    style=style,
	    style_order=plot_meta.style_order,
	    ax=ax,
	    palette=plot_meta.palette,
	    legend=True)

	ax.set_title(f'Filtered Samples of {column!r} {"w" if interpolate else "wo"} interp.')
	ax.set_xlabel('index')
	ax.grid(True)

	if not in_ax:
		add_scrollable_legend(ax)


def iterate_plotting_info(
    data: UngroupedData,
    columns,  # : str | tuple[str]
    split_groups: bool,
    split_columns: bool,
    in_axs,
    additional_gps=[],
    grid: bool = True,
    xlog: int = None,
    ylog: int = None,
    xflip: bool = False,
    yflip: bool = False,
):
	''' wrapper function to prepeare pandas dataframe to interact nicely with the seaborn potting framework '''

	columns = tuple_list(columns)

	# i like this one
	palette = {'palette': 'colorblind'}

	dfs = data.dfs

	# get nameing stuff
	group_names = [','.join(map(str, data.ctx.get_group_id(df))) for df in data.dfs]
	value_name = ','.join(columns)

	if not split_groups:
		dfs = [pd.concat(data.dfs)]
		group_names = ['All']

	n_rows = len(columns) if split_columns else 1

	# this size was fitting for desktop environemnts
	axss = [
	    plt.subplots(n_rows, 1, sharex=True, sharey=False, squeeze=False, figsize=(7, 3.5 * max(n_rows, 2)))[1][:, 0]
	    for _ in dfs
	]
	#axss = plt.subplots(n_rows, len(data.dfs), sharex='col', squeeze=False, figsize=(15*len(data.dfs), 10))[1].T

	for df, gn, axs in zip(dfs, group_names, axss):

		gps = []

		# if we are not splitting the groups we have to mark them
		if not split_groups:
			gps.extend(data.ctx.groups)

		# generate the data for the selected columns
		df = df.melt(id_vars=gps + additional_gps, var_name='Groups', value_vars=columns, ignore_index=False)

		# if we are not splitting the columns we have to mark them
		# also if we don't have any markings ... add some color and mark the columns
		if not split_columns or len(gps) == 0:
			gps.append('Groups')

		# generate hue and style
		df, hue = to_hue2(df, gps)
		df, style = to_style2(df, gps)

		def finalize_ax(ax):
			if grid:
				ax.grid(True)
			legend = ax.get_legend()
			if legend:
				legend.set_draggable(True)
				sns.move_legend(ax, loc='upper right')

			if xlog:
				ax.set_xscale('symlog', base=xlog)

			if ylog:
				ax.set_yscale('symlog', base=ylog)

			if xflip:
				ax.invert_xaxis()

			if yflip:
				ax.invert_yaxis()

		if split_columns:
			for cn, ax in zip(columns, axs):
				yield df[df.Groups == cn].rename(columns={'value': cn}), gn, cn, hue, style, palette, ax
				finalize_ax(ax)
		else:
			yield df.rename(columns={'value': value_name}), gn, value_name, hue, style, palette, axs[0]
			finalize_ax(axs[0])


@command('plot_hist', grouped=False)
def plot_samples_histogram(
    data: UngroupedData,
    columns,  # : str | tuple[str],
    split_groups: bool = False,
    split_columns: bool = False,
    in_ax=None,
    stat='count',
    bins='auto',
    kde: bool = False,
    grid: bool = True,
    xlog: int = None,
    ylog: int = None,
    xflip: bool = False,
    yflip: bool = False,
):

	for df, group_name, column_name, hue, style, palette, ax in iterate_plotting_info(
	    data,
	    columns,
	    split_groups,
	    split_columns,
	    in_ax,
	    grid=grid,
	    xlog=xlog,
	    ylog=ylog,
	    xflip=xflip,
	    yflip=yflip,
	):

		ax.set_title(f'Histogram plot of {column_name!r} for {group_name!r}')

		ax = sns.histplot(
		    data=df,
		    x=column_name,
		    **hue,
		    **palette,
		    ax=ax,
		    bins=bins,
		    legend=True,
		    element='step',
		    stat=stat,
		    fill=True,
		    kde=kde,
		)


@command('plot_chist', grouped=False)
def plot_samples_counted_histogram(
    data: UngroupedData,
    columns,  #: str | tuple[str],
    split_groups: bool = False,
    split_columns: bool = False,
    in_ax=None,
    stat='count',
    bins='auto',
    kde: bool = False,
    grid: bool = True,
    xlog: int = None,
    ylog: int = None,
    xflip: bool = False,
    yflip: bool = False,
):

	for df, group_name, column_name, hue, style, palette, ax in iterate_plotting_info(
	    data,
	    columns,
	    split_groups,
	    split_columns,
	    in_ax,
	    grid=grid,
	    xlog=xlog,
	    ylog=ylog,
	    xflip=xflip,
	    yflip=yflip,
	):

		ax.set_title(f'Histogram plot of {column_name!r} for {group_name!r}')

		ax = sns.histplot(
		    data=df,
		    x=column_name,
		    **hue,
		    **palette,
		    ax=ax,
		    bins=bins,
		    legend=True,
		    element='step',
		    stat=stat,
		    discrete=1,
		    fill=True,
		    kde=kde,
		)


@command('plot_kde', grouped=False)
def plot_samples_kde(
    data: UngroupedData,
    columns,  #: str | tuple[str],
    split_groups: bool = False,
    split_columns: bool = False,
    in_ax=None,
    grid: bool = True,
    xlog: int = None,
    ylog: int = None,
    xflip: bool = False,
    yflip: bool = False,
):

	for df, group_name, column_name, hue, style, palette, ax in iterate_plotting_info(
	    data,
	    columns,
	    split_groups,
	    split_columns,
	    in_ax,
	    grid=grid,
	    xlog=xlog,
	    ylog=ylog,
	    xflip=xflip,
	    yflip=yflip,
	):

		ax.set_title(f'KDE plot of {column_name!r} for {group_name!r}')

		ax = sns.kdeplot(
		    data=df,
		    x=column_name,
		    **hue,
		    **palette,
		    ax=ax,
		    legend=True,
		)


@command('plot_line', grouped=False)
def plot_samples_line(
    data: UngroupedData,
    columns,  #: str | tuple[str],
    split_groups: bool = False,
    split_columns: bool = False,
    in_ax=None,
    grid: bool = True,
    xlog: int = None,
    ylog: int = None,
    xflip: bool = False,
    yflip: bool = False,
    markers: bool = False,
):

	for df, group_name, column_name, hue, style, palette, ax in iterate_plotting_info(
	    data,
	    columns,
	    split_groups,
	    split_columns,
	    in_ax,
	    grid=grid,
	    xlog=xlog,
	    ylog=ylog,
	    xflip=xflip,
	    yflip=yflip,
	):

		ax.set_title(f'Line plot of {column_name!r} for {group_name!r}')

		ax = sns.lineplot(
		    data=df,
		    x=df.index,
		    y=column_name,
		    **hue,
		    **palette,
		    **style,
		    ax=ax,
		    legend=True,
		    markers=markers,
		)


@command('plot_scatter', grouped=False)
def plot_samples_scatter(
    data: UngroupedData,
    columns,  #: str | tuple[str],
    split_groups: bool = False,
    split_columns: bool = False,
    in_ax=None,
    grid: bool = True,
    xlog: int = None,
    ylog: int = None,
    xflip: bool = False,
    yflip: bool = False,
):

	for df, group_name, column_name, hue, style, palette, ax in iterate_plotting_info(
	    data,
	    columns,
	    split_groups,
	    split_columns,
	    in_ax,
	    grid=grid,
	    xlog=xlog,
	    ylog=ylog,
	    xflip=xflip,
	    yflip=yflip,
	):

		ax.set_title(f'Scatter plot of {column_name!r} for {group_name!r}')

		ax = sns.scatterplot(
		    data=df,
		    x=df.index,
		    y=column_name,
		    **hue,
		    **style,
		    **palette,
		    ax=ax,
		    legend=True,
		)


@command('plot_pointplot', grouped=False)
def plot_samples_pointplot(
    data: UngroupedData,
    columns,  #: str | tuple[str],
    split_groups: bool = False,
    split_columns: bool = False,
    in_ax=None,
    grid: bool = True,
    xlog: int = None,
    ylog: int = None,
    xflip: bool = False,
    yflip: bool = False,
):

	for df, group_name, column_name, hue, style, palette, ax in iterate_plotting_info(
	    data,
	    columns,
	    split_groups,
	    split_columns,
	    in_ax,
	    grid=grid,
	    xlog=xlog,
	    ylog=ylog,
	    xflip=xflip,
	    yflip=yflip,
	):

		ax.set_title(f'Pointplot plot of {column_name!r} for {group_name!r}')

		ax = sns.pointplot(
		    data=df,
		    x=df.index,
		    y=column_name,
		    **hue,
		    **palette,
		    ax=ax,
		    legend=True,
		    errorbar=('pi', 95),
		)


@command('plot_violin', grouped=False)
def plot_samples_violin(
    data: UngroupedData,
    columns,  #: str | tuple[str],
    split_groups: bool = False,
    split_columns: bool = False,
    in_ax=None,
    grid: bool = True,
    xlog: int = None,
    ylog: int = None,
    xflip: bool = False,
    yflip: bool = False,
):

	for df, group_name, column_name, hue, style, palette, ax in iterate_plotting_info(
	    data,
	    columns,
	    split_groups,
	    split_columns,
	    in_ax,
	    grid=grid,
	    xlog=xlog,
	    ylog=ylog,
	    xflip=xflip,
	    yflip=yflip,
	):

		ax.set_title(f'Violin plot of {column_name!r} for {group_name!r}')

		ax = sns.violinplot(
		    data=df,
		    x=df.index,
		    y=column_name,
		    **hue,
		    **palette,
		    ax=ax,
		    legend=True,
		    fill=True,
		    inner='quart',
		    split=True,
		    gap=.1,
		    gridsize=200,
		)


@command('plot_boxed', grouped=False)
def plot_samples_boxed(
    data: UngroupedData,
    columns,  #: str | tuple[str],
    split_groups: bool = False,
    split_columns: bool = False,
    in_ax=None,
    grid: bool = True,
    xlog: int = None,
    ylog: int = None,
    xflip: bool = False,
    yflip: bool = False,
):

	for df, group_name, column_name, hue, style, palette, ax in iterate_plotting_info(
	    data,
	    columns,
	    split_groups,
	    split_columns,
	    in_ax,
	    grid=grid,
	    xlog=xlog,
	    ylog=ylog,
	    xflip=xflip,
	    yflip=yflip,
	):

		ax.set_title(f'Boxplot plot of {column_name!r} for {group_name!r}')

		ax = sns.boxplot(
		    data=df,
		    x=df.index,
		    y=column_name,
		    **hue,
		    **palette,
		    ax=ax,
		    legend=True,
		    whis=(0, 100),
		)


@command('plot_line_stats', grouped=False)
def plot_samples_line_stats(
    data: UngroupedData,
    columns,  #: str | tuple[str],
    split_groups: bool = False,
    split_columns: bool = False,
    in_ax=None,
    grid: bool = True,
    xlog: int = None,
    ylog: int = None,
    xflip: bool = False,
    yflip: bool = False,
    parametric: bool = True,
):

	for df, group_name, column_name, hue, style, palette, ax in iterate_plotting_info(
	    data,
	    columns,
	    split_groups,
	    split_columns,
	    in_ax,
	    grid=grid,
	    xlog=xlog,
	    ylog=ylog,
	    xflip=xflip,
	    yflip=yflip,
	):

		ax.set_title(f'Lineplot plot of {column_name!r} for {group_name!r}')

		estimator, errorbar = ('mean', ('sd', 1)) if parametric else ('median', ('pi', 95))

		sns.lineplot(
		    data=df,
		    x=df.index,
		    y=column_name,
		    **hue,
		    **style,
		    **palette,
		    ax=ax,
		    legend=True,
		    estimator=estimator,
		    errorbar=errorbar,
		)


@command('plot_line_stats_eb', grouped=False)
def plot_line_stats_eb(
    data: UngroupedData,
    columns,  #: str | tuple[str],
    split_groups: bool = False,
    split_columns: bool = False,
    in_ax=None,
    grid: bool = True,
    xlog: int = None,
    ylog: int = None,
    xflip: bool = False,
    yflip: bool = False,
):

	bounds = sum([[c + '_l', c + '_u'] for c in tuple_list(columns)], [])

	for df, group_name, column_name, hue, style, palette, ax in iterate_plotting_info(
	    data,
	    columns,
	    split_groups,
	    split_columns,
	    in_ax,
	    additional_gps=bounds,
	    grid=grid,
	    xlog=xlog,
	    ylog=ylog,
	    xflip=xflip,
	    yflip=yflip,
	):

		ax.set_title(f'Lineplot plot of {column_name!r} for {group_name!r}')
		ax = sns.lineplot(data=df, x=df.index, y=column_name, **hue, **palette, ax=ax, legend=True)

		for ho, l in zip(hue['hue_order'], ax.get_lines()):
			x = df[df[hue['hue']] == ho]
			ax.fill_between(x.index, column_name + '_l', column_name + '_u', data=x, alpha=0.2, facecolor=l.get_c())


@command('show', grouped=False)
def plot_show(data: UngroupedData):
	from matplotlib.pyplot import show
	show()
	return data.dfs


@command('show_web', grouped=False)
def plot_show_web(data: UngroupedData):
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	mpl.use('WebAgg')
	mpl.rcParams['webagg.open_in_browser'] = False
	print('use local port forwarding to view remote: ssh lab22 -L 8988:127.0.0.1:8988')
	plt.show()
	exit(0)


def plot_similarity_matrix(df, meta, column, name, in_ax=None):
	plot_meta = get_plot_meta(df, meta)

	ax = in_ax or create_new_axis(f'{name} plot of {column!r}')

	ax.set_title(f'{name} plot of {column!r}')
	# ax = sns.heatmap(data=df, vmin=0, vmax=1, annot=False, ax=ax)
	im = ax.imshow(np.array(df), cmap=plot_meta.palette, interpolation='none', aspect='auto', vmin=0, vmax=1)

	ax.get_figure().colorbar(im, ax=ax)

	ax.set_yticks(np.arange(len(df.index)))
	ax.set_xticks(np.arange(len(df.index)))

	ax.set_yticklabels(df.index)
	ax.set_xticklabels(df.index, ha='right')

	ax.tick_params(axis='x', rotation=45)
	ax.tick_params(axis='both', labelsize=LABELSIZE)


def plot_group_stats(df, meta, column, in_axs=None):
	rda.utils.valid_columns(df, [column])

	plot_meta = get_plot_meta(df, meta)

	axs = in_axs or create_new_axis(f'Stats plot of {column!r}', 1, 2)

	df_stats = rda.utils.groupby(df, meta.groups)[column].agg(
	    ('count', 'min', 'mean', 'median', 'max', 'std', 'sem')).reset_index()
	print('Stats:')
	print(df_stats)

	for ax, name in zip(axs, ['Mean', 'Median']):

		x = to_str(df_stats, meta.groups)

		ax.set_title(name)
		# breakpoint()
		ax = sns.scatterplot(
		    data=df_stats,
		    x=x,
		    y=name.lower(),
		    hue=to_hue(df_stats, meta.groups),
		    hue_order=plot_meta.hue_order,
		    style=to_style(df_stats, meta.groups),
		    style_order=plot_meta.style_order,
		    palette=plot_meta.palette,
		    ax=ax,
		    legend=True)
		# ax = sns.boxenplot(data=df, x=plot_meta.hue, order=plot_meta.hue_order, y=column, hue=plot_meta.hue, hue_order=plot_meta.hue_order, ax=ax, palette=plot_meta.palette)
		ax.tick_params(axis='x', rotation=45)
		ax.tick_params(axis='x', labelsize=LABELSIZE)
		ax.set_ylabel('')
		ax.grid(True)

		if not in_axs:
			add_scrollable_legend(ax)


@command('plot_heatmap', grouped=False)
def plot_heatmap(data: UngroupedData,
                 columns: str,
                 x_col: str,
                 y_col: str,
                 normalize=False,
                 transpose=False,
                 rotate_labels=False,
                 save=False,
                 axs=None,
                 annot=True,
                 robust=False,
                 store=''):

	for df, group_name, column_name, hue, style, palette, ax in iterate_plotting_info(
	    data,
	    columns,
	    split_groups=True,
	    split_columns=False,
	    in_axs=None,
	    additional_gps=[x_col, y_col],
	    grid=False,
	):

		#df = df.groupby([x_col, y_col]).reset_index()
		df = df.pivot(index=x_col, columns=y_col, values=column_name)

		if normalize and df.shape[0] != 1:
			df = (df - np.nanmin(df)) / (np.nanmax(df) - np.nanmin(df))

		if transpose:
			df = df.T

		annot_kws = {'rotation': 'vertical'} if rotate_labels else {}

		# axs[i].set_title(f'{meta.groups}={value} with {str(stat)}')

		ax.set_title(f'Heatmap of {column_name!r} for {group_name!r} with  {x_col!r} {y_col!r}')
		ax = sns.heatmap(
		    data=df,
		    robust=robust,
		    annot=annot or (df.shape[0] <= 10),
		    fmt='.4f',
		    ax=ax,
		)

		heatmap = df.melt(var_name=y_col, value_name=column_name, ignore_index=False)

		heatmap = heatmap.sort_values([y_col, x_col]).reset_index()

		if store != '':
			heatmap.to_csv(group_name + store, index=False)


class PlotMeta:

	def __init__(self, df, groups):
		hue = to_hue(df, groups)
		style = to_style(df, groups)

		df[hue.name] = hue
		df[style.name] = style

		self.groups = []
		self.groups.extend(groups)

		self.hue = hue.name
		self.style = style.name

		self.hue_order = hue.unique()
		self.style_order = style.unique()

		self.palette = 'colorblind'  #'rocket_r'

		try:
			self.hue_order.sort()
		except:
			pass

		try:
			self.style_order.sort()
		except:
			pass

	def get_hue(self):
		return {'hue': self.hue, 'hue_order': self.hue_order}

	def get_style(self):
		return {'style': self.style, 'style_order': self.style_order}


def get_plot_meta(df, meta):
	if 'plot' not in meta.module_data:
		meta.module_data['plot'] = PlotMeta(df, meta.groups)

	# if meta.module_data['plot'].groups != meta.groups:
	meta.module_data['plot'] = PlotMeta(df, meta.groups)

	return meta.module_data['plot']


def plot_overview(df, meta, column):

	fig = plt.figure(figsize=(16, 9))
	fig.canvas.manager.set_window_title(f'Overview plot of {column!r}')
	plt.subplots_adjust(hspace=0.5)

	gs = gridspec.GridSpec(3, 4)

	axs = [
	    plt.subplot(gs[0, 0:2]),
	    plt.subplot(gs[0, 2:4]),
	    plt.subplot(gs[1, 0:2]),
	    plt.subplot(gs[1, 2:4]),
	    plt.subplot(gs[2, 0]),
	    plt.subplot(gs[2, 1]),
	    plt.subplot(gs[2, 2:3]),
	    plt.subplot(gs[2, 3:4]),
	]

	KS, MW = calculate_similarity_matricies(df, meta.groups, column)

	plot_samples_scatter(df, meta, [column], [axs[0]])
	plot_samples_filtered(df, meta, column, axs[1])
	plot_samples_histogram(df, meta, column, axs[2])
	plot_samples_kde(df, meta, column, axs[3])

	plot_similarity_matrix(MW, meta, column, 'MW', axs[4])

	plot_similarity_matrix(KS, meta, column, 'KS', axs[5])

	plot_group_stats(df, meta, column, axs[6:8])

	add_scrollable_legend(axs[0], loc='upper left')
	add_scrollable_legend(axs[1], loc='upper right')
	add_scrollable_legend(axs[2], loc='upper right')
	add_scrollable_legend(axs[3], loc='upper left')
	add_scrollable_legend(axs[6], loc='upper right')
	add_scrollable_legend(axs[7], loc='upper left')

	# axs[1].set_xticklabels(labels=axs[1].get_xticklabels(), ha='right')
	# axs[3].set_xticklabels(labels=axs[3].get_xticklabels(), ha='right')

	# if save:
	#    fig.savefig(f'{file}.{column}_overall.png', dpi=400)
	#    lfig.savefig(f'{file}.{column}_legend.png', dpi=400)

	# import tikzplotlib
	#   tikzplotlib.save(figure=fig,filepath='test.tex')
