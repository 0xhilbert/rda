import os

os.environ["MPLBACKEND"] = "WebAgg"
os.environ["CUPY_CACHE_SAVE_CUDA_SOURCE"] = "1"
os.environ["CUPY_DUMP_CUDA_SOURCE_ON_ERROR"] = "1"

import fire

from matplotlib.pyplot import show

import matplotlib as mpl
import matplotlib.style as mplstyle

mplstyle.use("fast")

mpl.rcParams["webagg.open_in_browser"] = False
mpl.rcParams["figure.max_open_warning"] = 1000

from rda.cli.command import RapidDataAnalysis

# commands
import rda.utils
import rda.power.power
import rda.plot.plot
import rda.processing.processing
import rda.cli.basic
import rda.cli.shorthands


def main():
	fire.Fire(RapidDataAnalysis, name="RapidDataAnalysis")
	show()


if __name__ == "__main__":
	main()
