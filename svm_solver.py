import numpy as np
import pickle, sys
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

COLORS = ['red', 'blue']

def read_data(f):
    with open(f, 'rb') as f:
        data = pickle.load(f)
    x, y = data[0], data[1]
    return x, y

def fit(x, y):
    NUM = x.shape[0]
    DIM = x.shape[1]
    # we'll solve the dual
    # obtain the kernel
    K = y[:, None] * x
    K = np.dot(K, K.T)
    P = matrix(K)
    q = matrix(-np.ones((NUM, 1)))
    G = matrix(-np.eye(NUM))
    h = matrix(np.zeros(NUM))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas

def plot_data_with_labels(x, y, ax):
    unique = np.unique(y)
    for li in range(len(unique)):
        x_sub = x[y == unique[li]]
        ax.scatter(x_sub[:, 0], x_sub[:, 1], c = COLORS[li])

def plot_separator(ax, w, b):
    slope = -w[0] / w[1]
    intercept = -b / w[1]
    x = np.arange(0, 6)
    ax.plot(x, x * slope + intercept, 'k-')

if __name__ == '__main__':
    x, y = read_data(sys.argv[1])
    # fit svm classifier
    alphas = fit(x, y)

    # get weights
    w = np.sum(alphas * y[:, None] * x, axis = 0)
    # get bias
    cond = (alphas > 1e-4).reshape(-1)
    b = y[cond] - np.dot(x[cond], w)
    bias = b[0]

    # normalize
    norm = np.linalg.norm(w)
    w, bias = w / norm, bias / norm

    # show data and w
    fig, ax = plt.subplots()
    plot_separator(ax, w, bias)
    plot_data_with_labels(x, y, ax)
    plt.show()
