import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import rolling_window
from argparse import ArgumentParser


def build_parser():
    parser = ArgumentParser(description='Plot Args.')
    parser.add_argument('--model_num', type=str, required=True,
                        help='Model number for accessing correct log folder')
    parser.add_argument('--name', type=str, default='loss.png',
                        help='File name for saving to plot dir')
    parser.add_argument('--mode', type=str, default='loss',
                        help='Whether to plot loss or validation')
    return parser


def main(args):
    exp = args.model_num
    name = args.name
    loss = True if args.mode == 'loss' else False

    plot_dir = './plots/'
    logs_dir = './logs/'
    logs_dir += 'exp_{}/'.format(exp)
    plot_dir += 'exp_{}/'.format(exp)

    if loss:
        filename = logs_dir + 'train.csv'
    else:
        filename = logs_dir + 'valid.csv'
    df = pd.read_csv(filename, header=None, names=['iter', 'metric'])
    metric = df['metric']

    fig, ax = plt.subplots(figsize=(15, 8))
    rolling_mean = np.mean(rolling_window(metric, 50), 1)
    rolling_std = np.std(rolling_window(metric, 50), 1)
    plt.plot(range(len(rolling_mean)), rolling_mean, alpha=0.98, linewidth=0.9)
    plt.fill_between(
        range(len(rolling_std)),
        rolling_mean-rolling_std,
        rolling_mean+rolling_std,
        alpha=0.5
    )
    title = 'Train Loss' if loss else 'Valid Acc'
    xtitle = 'Loss' if loss else 'Acc'
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(xtitle)
    plt.grid()
    plt.tight_layout()
    plt.savefig(plot_dir + name, format='png', dpi=300)


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    main(args)
