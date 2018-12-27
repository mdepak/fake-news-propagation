from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


def perform_t_test(samples1, samples2):
    [t_val, p_val] = stats.ttest_ind(samples1, samples2, equal_var=False)
    print("t-Statistic value : {}".format(t_val))
    print("p - value : {}".format(p_val))


def plot_normal_distributions(samples1, samples2):
    fit1 = stats.norm.pdf(samples1, np.mean(samples1), np.std(samples1))
    fit2 = stats.norm.cdf(samples2, np.mean(samples2), np.std(samples2))

    plt.plot(sorted(samples1), fit1, 'red')
    plt.plot(sorted(samples2), fit2, 'blue')
    plt.show()


def get_box_plots(samples1, samples2, title = None):
    all_data = [samples1, samples2]
    labels = ['Fake', 'Real']

    # rectangular box plot
    bplot1 = plt.boxplot(all_data,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks
    plt.title(title)

    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen']
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    # adding horizontal grid lines
    # plt.xlabel()
    # plt.ylabel()

    plt.show()
