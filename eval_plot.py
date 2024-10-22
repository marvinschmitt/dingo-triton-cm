import sys
import matplotlib.pyplot as plt
from dingo.gw.result import Result
import argparse
from chainconsumer import ChainConsumer
import numpy as np
from scipy.special import erfinv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and plot GW results.")
    parser.add_argument('method', type=str, help='The method to use for evaluation.')
    return parser.parse_args()

args = parse_args()

print(f"********** Plotting results for {args.method} method. **********")

params=['luminosity_distance', 'geocent_time', 'ra', 'dec']
result_method = Result(file_name=f"training/{args.method}/eval/result/GW150914_data0_1126259462-421_sampling.hdf5").samples[params]
result_fmpe = Result(file_name=f"training/fmpe/eval/result/GW150914_data0_1126259462-421_sampling.hdf5").samples[params]

print(Result(file_name=f"training/{args.method}/eval/result/GW150914_data0_1126259462-421_sampling.hdf5").samples)

c = ChainConsumer()
c.add_chain(result_method, weights=None, color="#FFA500", name=args.method)
c.add_chain(result_fmpe, weights=None, color="#006400", name='fmpe')

n = 2

c.configure(
    linestyles=["-"] * n,
    linewidths=[1.5] * n,
    sigmas=[np.sqrt(2) * erfinv(x) for x in [0.9]],
    shade=[False] * n,
    shade_alpha=0.3,
    bar_shade=False,
    label_font_size=10,
    tick_font_size=10,
    usetex=False,
    legend_kwargs={"fontsize": 18},
    kde=0.7,
)
c.plotter.plot(filename=f"training/{args.method}/eval/result/GW150914_corner.pdf")

# corner.corner(result.samples, labels=params, show_titles=True)
# plt.savefig(f"training/{args.method}/eval/result/GW150914_corner.pdf")
# result.plot_corner(filename=f"training/{args.method}/eval/result/GW150914_corner.pdf", parameters=params)

print(f"********** Done plotting results for {args.method} method. **********")