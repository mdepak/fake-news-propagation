from textwrap import wrap

from scipy import stats
import numpy as np

import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd


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


def get_box_plots(samples1, samples2, save_folder, title=None, file_name=None):
    all_data = [samples1, samples2]
    labels = ['Fake', 'Real']
    # plt.box(None)

    font = {'family': 'normal',
            'weight': 'semibold',
            'size': 13}
    #
    matplotlib.rc('font', **font)

    # plt.xlabel('l', fontsize=18)
    # plt.ylabel('ylabel', fontsize=16)
    plt.tight_layout()
    plt.figure(figsize=(1.5, 4))

    fig = plt.figure(1, figsize=(1, 3), frameon=False)

    ax1 = fig.add_subplot(111)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # ax.set_aspect(2)
    # rectangular box plot
    bplot1 = ax1.boxplot(all_data,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels,  # will be used to label x-ticks
                         showfliers=False,
                         positions=[0, 0.5])

    # ax2 = fig.add_subplot(121)
    # # ax.set_aspect(2)
    # # rectangular box plot
    # bplot2 = ax2.boxplot(all_data,
    #                      vert=True,  # vertical box alignment
    #                      patch_artist=True,  # fill with color
    #                      labels=labels)  # will be used to label x-ticks

    # plt.title(title)
    # title = ax1.set_title("\n".join(wrap(title,50)), fontdict={'fontweight': 'semibold'})
    [t_val, p_val] = stats.ttest_ind(samples1, samples2, equal_var=True)

    if p_val > 0.05:
        title = ax1.set_title(file_name, fontdict={'fontweight': 'bold', 'fontsize': 16})
    else:
        ax1.set_title(r'' + file_name + ' $\mathbf{^{*}}$', fontdict={'fontweight': 'bold', 'fontsize': 16})
    # fill with colors

    # ax1.set_xlim(1, 0)

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


def get_box_plots_mod(samples1, samples2, save_folder, title=None, file_name=None):
    all_data = np.transpose(np.array([samples1, samples2]))
    labels = ['Fake', 'Real']
    df = pd.DataFrame(all_data, columns=labels)
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import pyplot

    fig, ax = pyplot.subplots(figsize=(3, 3.5))

    my_pal = {"Fake": "pink", "Real": "lightblue", }

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax = sns.boxplot(data=df, width=0.3, palette=my_pal,  showfliers=False)

    colors = ['pink', 'lightblue']
    for idx,  patch in enumerate(ax.artists):
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor(colors[idx])

    [t_val, p_val] = stats.ttest_ind(samples1, samples2, equal_var=True)

    if p_val > 0.05:
        title = plt.title(file_name, fontdict={'fontweight': 'bold', 'fontsize': 16})
    else:
        plt.title(r'' + file_name + ' $\mathbf{^{*}}$', fontdict={'fontweight': 'bold', 'fontsize': 16})

    plt.savefig('{}/{}.png'.format(save_folder, file_name),bbox_inches="tight")

    plt.show()

    # plt.box(None)

    return
    font = {'family': 'normal',
            'weight': 'semibold',
            'size': 13}
    #
    matplotlib.rc('font', **font)

    # plt.xlabel('l', fontsize=18)
    # plt.ylabel('ylabel', fontsize=16)
    plt.tight_layout()
    plt.figure(figsize=(1.5, 4))

    fig = plt.figure(1, figsize=(1, 3), frameon=False)

    ax1 = fig.add_subplot(111)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # ax.set_aspect(2)
    # rectangular box plot
    bplot1 = ax1.boxplot(all_data,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels,  # will be used to label x-ticks
                         showfliers=False,
                         positions=[0, 0.5])

    # ax2 = fig.add_subplot(121)
    # # ax.set_aspect(2)
    # # rectangular box plot
    # bplot2 = ax2.boxplot(all_data,
    #                      vert=True,  # vertical box alignment
    #                      patch_artist=True,  # fill with color
    #                      labels=labels)  # will be used to label x-ticks

    # plt.title(title)
    # title = ax1.set_title("\n".join(wrap(title,50)), fontdict={'fontweight': 'semibold'})
    [t_val, p_val] = stats.ttest_ind(samples1, samples2, equal_var=True)

    if p_val > 0.05:
        title = ax1.set_title(file_name, fontdict={'fontweight': 'bold', 'fontsize': 16})
    else:
        ax1.set_title(r'' + file_name + ' $\mathbf{^{*}}$', fontdict={'fontweight': 'bold', 'fontsize': 16})
    # fill with colors

    # ax1.set_xlim(1, 0)

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


if __name__ == "__main__":
    get_box_plots_mod(np.random.rand(200, ), np.random.rand(200, ),
                      "/Users/deepak/Desktop/DMML/GitRepo/FakeNewsPropagation",
                      "T10", "T10")
