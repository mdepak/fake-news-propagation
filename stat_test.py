from textwrap import wrap

from scipy import stats
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def perform_t_test(samples1, samples2):
    [t_val, p_val] = stats.ttest_ind(samples1, samples2, equal_var=True)
    print("t-Statistic value : {}".format(t_val))
    print("p - value : {}".format(p_val))
    print("=====================================")


def plot_normal_distributions(samples1, samples2):
    fit1 = stats.norm.pdf(samples1, np.mean(samples1), np.std(samples1))
    fit2 = stats.norm.cdf(samples2, np.mean(samples2), np.std(samples2))

    plt.plot(sorted(samples1), fit1, 'red')
    plt.plot(sorted(samples2), fit2, 'blue')
    plt.show()


def get_box_plots(samples1, samples2, save_folder,  title = None, file_name = None):
    all_data = [samples1, samples2]
    labels = ['Fake', 'Real']
    # plt.box(None)
    fig = plt.figure(1, figsize=(4.5, 4))
    ax1 = fig.add_subplot(111)

    # ax.set_aspect(2)
    # rectangular box plot
    bplot1 = ax1.boxplot(all_data,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels, showfliers=False)  # will be used to label x-ticks

    # ax2 = fig.add_subplot(121)
    # # ax.set_aspect(2)
    # # rectangular box plot
    # bplot2 = ax2.boxplot(all_data,
    #                      vert=True,  # vertical box alignment
    #                      patch_artist=True,  # fill with color
    #                      labels=labels)  # will be used to label x-ticks

    # plt.title(title)
    # title = ax1.set_title("\n".join(wrap(title,50)), fontdict={'fontweight': 'semibold'})
    title = ax1.set_title(file_name, fontdict={'fontweight': 'bold', 'fontsize':15})
    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen']
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    # fmt = matplotlib.ticker.ScalarFormatter(useOffset=False)
    # fmt.set_scientific(False)
    # ax1.yaxis.set_major_formatter(fmt)

    # ax1.set_yticklabels(ax1.get_yticks())

    # adding horizontal grid lines
    # plt.xlabel()
    # plt.ylabel()

    fig.savefig('{}/{}.png'.format(save_folder, file_name))
    fig.show()
    plt.close()
