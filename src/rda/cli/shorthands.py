from rda.cli.command import (
    RapidDataAnalysis,
    GroupedData,
    UngroupedData,
    command,
    tuple_list,
    add_groups,
)


# open file and init power stuff
@command("pw_i", grouped=False)
def power_input(
    data: UngroupedData,
    file: str = '',
    bits: int = 8,
    nrows: int = None,
    wstride: int = 1,
    wcount: int = -1,
    machine: str = "mlab07",
    diff: bool = True,
    no_expand: bool = False,
    blocks=None,
    filter_unused='NONE',
    add_file_info: bool = False,
):
	x = RapidDataAnalysis(data.dfs, data.ctx)

	# open file
	if file != '':
		x.i(
		    file,
		    nrows=nrows,
		    add_file_info=add_file_info,
		)

	# init power
	x.power_init(
	    machine=machine,
	    bits=bits,
	    wstride=wstride,
	    wcount=wcount,
	    no_diff=not diff,
	    no_expand=no_expand,
	    filter_unused=filter_unused,
	)

	# convert byte string to real string
	x.decode_str('Exp')

	# convert to time or block indexed data if requestd
	if blocks is not None:
		if isinstance(blocks, str):
			x.to_time('time')
			x.idx('time')
			x.blocks(blocks)
		elif isinstance(blocks, int):
			x.blocks(blocks)

	return x.dfs


# helper function to retrieve measurement columns
def ri_fields(x, name):
	if f'D{name}' in x.dfs[0]:
		return [f'R{name}', f'I{name}']
	else:
		return [f'{name}']


# helper function to retrieve measurement columns
def d_fields(x, name):
	if f'D{name}' in x.dfs[0]:
		return [f'D{name}']
	else:
		return [f'{name}']


@command("pw_sh", grouped=False)
def power_signal_histograms(data: UngroupedData, bins=200):
	x = RapidDataAnalysis(data.dfs, data.ctx)

	kw = {"split_groups": 0, "split_columns": 1, "bins": bins}

	x.plot_hist(ri_fields(x, 'Energy'), **kw)
	x.plot_hist(ri_fields(x, 'EnergyPP0'), **kw)
	x.plot_hist(ri_fields(x, 'PowerPP0'), **kw)

	x.plot_hist(ri_fields(x, 'Ticks'), **kw)
	x.plot_hist(ri_fields(x, 'Freq'), **kw)
	x.plot_hist(ri_fields(x, 'PState'), **kw)
	x.plot_hist(ri_fields(x, 'Temp'), **kw)
	x.plot_hist(ri_fields(x, 'Volt'), **kw)

	x.plot_hist(d_fields(x, 'Energy'), **kw)
	x.plot_hist(d_fields(x, 'EnergyPP0'), **kw)
	x.plot_hist(d_fields(x, 'PowerPP0'), **kw)

	x.plot_hist(['la'], **kw)


def plot_lines_all(x: RapidDataAnalysis, kw):

	x.plot_line_stats(ri_fields(x, 'Energy'), **kw)
	x.plot_line_stats(ri_fields(x, 'EnergyPP0'), **kw)
	x.plot_line_stats(ri_fields(x, 'PowerPP0'), **kw)

	x.plot_line_stats(ri_fields(x, 'Ticks'), **kw)
	x.plot_line_stats(ri_fields(x, 'Freq'), **kw)
	x.plot_line_stats(ri_fields(x, 'PState'), **kw)
	x.plot_line_stats(ri_fields(x, 'Temp'), **kw)
	x.plot_line_stats(ri_fields(x, 'Volt'), **kw)

	x.plot_line_stats(ri_fields(x, 'temp_a'), **kw)
	x.plot_line_stats(ri_fields(x, 'temp_b'), **kw)

	x.plot_line_stats(d_fields(x, 'Energy'), **kw)
	x.plot_line_stats(d_fields(x, 'EnergyPP0'), **kw)
	x.plot_line_stats(d_fields(x, 'PowerPP0'), **kw)

	x.plot_line_stats(['la'], **kw)


@command("pw_sot", grouped=False)
def power_signals_over_time(data: UngroupedData):
	x = RapidDataAnalysis(data.dfs, data.ctx)

	kw = {"split_groups": 1, "split_columns": 0}

	plot_lines_all(x, kw)


@command("pw_sox", grouped=False)
def power_signals_over_x(data: UngroupedData, column: str):
	x = RapidDataAnalysis(data.dfs, data.ctx)

	kw = {"split_groups": 1, "split_columns": 0}

	# we index the data frame to change x axis
	x.idx(column)

	plot_lines_all(x, kw)


@command('move_vx_gx')
def move_vx_gx(data: GroupedData):

	def move(df):
		index = int(df.iloc[0].Exp[1:])

		df.v000 = df[f'v{index:03d}']
		df.g000 = df[f'g{index:03d}']

		return df

	return move(data.df)


# reduce the same measurements in the EREP
@command('pw_reduce', grouped=False)
def power_reduce(data: UngroupedData, function='mean'):
	x = RapidDataAnalysis(data.dfs, data.ctx)

	columns = ['la']

	columns += d_fields(x, 'Energy')
	columns += d_fields(x, 'EnergyPP0')
	columns += d_fields(x, 'EnergyDRAM')

	columns += d_fields(x, 'Power')
	columns += d_fields(x, 'PowerPP0')
	columns += d_fields(x, 'PowerDRAM')

	columns += d_fields(x, 'Freq')
	columns += d_fields(x, 'Ticks')
	columns += d_fields(x, 'Temp')

	# filter non present columns
	columns = tuple([c for c in columns if c in data.dfs[0]])

	x.reduce_g(
	    'ERep',
	    columns,
	    function,
	    keep_first=True,
	)

	return x.dfs


@command("pw_map", grouped=False)
def power_leakage_map(data: UngroupedData, field: str, abs=False, annot=True, extended=False):
	x = RapidDataAnalysis(data.dfs, data.ctx)

	models = ["hd_all()"]

	if extended:
		models.append("hd_self()")

	x.power_models(*models)

	x.power_fit(field, independent_coefficients=True)
	x.print()

	if abs:
		x.e("x.rho=x.rho.abs()")

	with add_groups(x, "type"):
		x.dropna(["rho"])
		x.plot_heatmap("rho", "x", "y", annot=annot, robust=False)
		x.plot_heatmap("coef", "x", "y", annot=annot, robust=False)


@command("pw_cpa", grouped=False)
def power_extended_cpa(
    data: UngroupedData,
    field: str,
    normalization='POST',
    gpu=True,
    fit=False,
    step=0,
    blocked=False,
):
	x = RapidDataAnalysis(data.dfs, data.ctx)

	# if we want to use a fitted model use this
	if fit:
		x.save(0)
		x.sel("x.Exp=='train'")
		x.power_fit(field, update_model=True, independent_coefficients=False)
		x.print()
		x.load(0)

	# remove training data
	x.sel("x.Exp!='train'")

	# perform the CPA
	x.power_cpa(
	    field,
	    constant_secret_values=False,
	    normalization=normalization,
	    use_gpu=gpu,
	    step=step,
	    blocked=blocked,
	)

	# plot iterative results if used them
	if step != 0:
		x.idx('N')
		x.g(('model', 'est'))
		x.plot_scatter('rank')
		x.plot_scatter('rank', ylog=2)


@command("pw_running", grouped=False)
def power_running(data: UngroupedData, n=10000):
	x = RapidDataAnalysis(data.dfs, data.ctx)

	cols = []

	cols += ri_fields(x, 'Energy')
	cols += ri_fields(x, 'EnergyPP0')
	cols += ri_fields(x, 'Ticks')
	cols += ri_fields(x, 'Freq')
	cols += ri_fields(x, 'Volt')
	cols += ri_fields(x, 'Temp')

	x.get_stats(cols, number_samples=n)

	for c in cols:
		x.plot_line(
		    tuple([f"{c}_{s}" for s in [
		        "mean",
		        "p05",
		        "p95",
		        "median",
		        "range",
		        "std",
		        "min",
		        "max",
		    ]]),
		    split_groups=1,
		    split_columns=1,
		)


@command("pw_cot_running", grouped=False)
def correlation_over_time_running(data: UngroupedData, field: str, n=75000):
	x = RapidDataAnalysis(data.dfs, data.ctx)

	raise RuntimeWarning('not sure of this currently works - use blocked version')

	x.power_fit(field, 0, all_ones=True, window=n)

	x.blocks(n)

	x.reduce_g("block", ["rho"], "min")
	x.plot_line_stats(["rho"], split_groups=1, split_columns=0)


@command("pw_cox", grouped=False)
def correlation_over_x(data: UngroupedData, field: str, index: str, split_groups=True, split_columns=False):
	x = RapidDataAnalysis(data.dfs, data.ctx)

	with add_groups(x, index):
		x.power_fit(field, 0, all_ones=True)

	x.idx(index)

	with add_groups(x, "model"):
		x.plot_line_stats(["rho", "rho_l", "rho_u"], split_groups=split_groups, split_columns=split_columns)


@command("pw_cot", grouped=False)
def correlation_over_time(data: UngroupedData, field: str, split_groups=True, split_columns=False):
	x = RapidDataAnalysis(data.dfs, data.ctx)

	x.pw_cox(field, 'block', split_groups=split_groups, split_columns=split_columns)


@command("pw_coe", grouped=False)
def correlation_over_experimentrepeat(data: UngroupedData, field: str, split_groups=True, split_columns=False):
	x = RapidDataAnalysis(data.dfs, data.ctx)

	x.pw_cox(field, 'ERep', split_groups=split_groups, split_columns=split_columns)
