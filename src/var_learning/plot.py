import os
import matplotlib.pyplot as plt


def hist_one(x1, bins, range, name, xlabel, ylabel='Count'):
    fig, ax = plt.subplots()
    ax.hist(x1, bins=bins, range=range, color='blue')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.show()
    os.makedirs('plots/', exist_ok=True)
    fig.savefig('plots/' + name + '.pdf')
    plt.close(fig)


def hist_two(x1, x2, bins, range, name, xlabel, ylabel='Count',
             label1='1', label2='2'):
    fig, ax = plt.subplots()
    ax.hist(x1, bins=bins, label=label1, range=range, color='blue', alpha=0.5)
    ax.hist(x2, bins=bins, label=label2, range=range, color='red', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.show()
    os.makedirs('plots/', exist_ok=True)
    fig.savefig('plots/' + name + '.pdf')
    plt.close(fig)


def plot_one(x1, y1, name, xlabel, ylabel='Count'):
    fig, ax = plt.subplots()
    ax.plot(x1, y1, range=range, color='blue')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.show()
    os.makedirs('plots/', exist_ok=True)
    fig.savefig('plots/' + name + '.pdf')
    plt.close(fig)


def plot_two(x1, y1, x2, y2, name, xlabel, ylabel='Count',
             label1='1', label2='2'):
    fig, ax = plt.subplots()
    ax.plot(x1, y1, label=label1, color='blue')
    ax.plot(x2, y2, label=label2, color='red')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.show()
    os.makedirs('plots/', exist_ok=True)
    fig.savefig('plots/' + name + '.pdf')
    plt.close(fig)
