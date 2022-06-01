import matplotlib.pyplot as plt
import numpy as np


def create_plot():
    # specify bar width
    bar_width = 0.3

    # specify font size
    fs = 18

    # specify bar heights
    bars1 = [1951099, 146417]
    bars2 = [1782717, 1557102]

    # specify x position
    x1 = [0, 1]

    # plot graph
    fig, ax = plt.subplots(1, 2, figsize=(18,10), sharey=True)
    fig.tight_layout(pad=12.0)
    ax[0].bar(x1, bars1, color=('#000000','#808080'))
    ax[1].bar(x1, bars2, color=('#000000','#808080'))
    ax[0].set_title('Class distribution: non-bankrupt and bankrupt firms', fontsize=fs, pad=41)
    ax[1].set_title('Class distribution: non-bankrupt and bankrupt firms\nwith SMOTEENN', fontsize=fs, pad=20)
    ax[0].set_ylabel('Number of firms', fontsize=fs)
    ax[0].set_yticks(np.arange(0, 2500000, 500000))
    ax[0].set_yticklabels(np.arange(0, 2500000, 500000), fontsize=fs)
    ax[0].set_xticks((0, 1))
    ax[0].set_xticklabels(('Non-Bankrupt', 'Bankrupt'), fontsize=fs)
    ax[1].set_ylabel('Number of firms', fontsize=fs)
    ax[1].set_yticks(np.arange(0, 2500000, 500000))
    ax[1].set_yticklabels(np.arange(0, 2500000, 500000), fontsize=fs)
    ax[1].set_xticks((0, 1))
    ax[1].set_xticklabels(('Non-Bankrupt', 'Bankrupt'), fontsize=fs)
    ax[1].yaxis.set_tick_params(labelbottom=True)
    plt.savefig('graphs/class_distribution.png', dpi=600)
    plt.show()

if __name__ == '__main__':
    create_plot()
    print('done')