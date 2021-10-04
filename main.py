import matplotlib.pyplot as plt
import numpy as np
from sympy import lambdify, factorial, cos, ln
from sympy.abc import x as sympy_x


class Lagrange:
    def __init__(self, n, x_f):
        """
        Lagrange polinom
        :param n:
        :param x_f: [[x,...,xn][f(x),...,f(xn)]]
        """
        self.n = n
        self.x_f = x_f

        self.l_j = lambda i, j, X, x: (X - x[j]) / (x[i] - x[j]) if i != j else 1
        self.l_i = lambda i, X: np.prod(np.array([self.l_j(i, part_index, X, self.x_f[0])
                                                  for part_index in range(self.n + 1)]))
        self.L = lambda X: np.sum(np.array([self.l_i(i, X) * self.x_f[1][i] for i in range(0, self.n + 1)]))


def interpolation(n, left, right, func, result_file):

    f = lambdify([sympy_x], func, "numpy")

    epsilon = 0.00006

    step = (right - left) / n

    x = np.arange(left, right + epsilon, step)
    y = f(x)

    x_f = np.array([x, y])

    lag = Lagrange(n, x_f)

    new_x = np.arange(left, right, ((right - left) / 1000))

    new_y = np.array([lag.L(x) for x in new_x])

    plt.plot(x, y, '--g', new_x, new_y, '-y')
    ax = plt.gca()
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    proisvodnaya = func.diff(sympy_x, n + 1)
    fs = lambdify([sympy_x], proisvodnaya, "numpy")

    with open(result_file, 'w') as out_file:
        out_file.write('\n'.join(
            [' '.join(
                [str(item[0]), str(item[1]), 'True' if item[0] <= item[1] else 'False'])
                for item in
                [
                    [abs(
                        new_y[index] - f(new_x[index])
                    ), abs(
                        (abs(
                            max([fs(x[prod_index])
                                 for prod_index in range(n + 1)]))
                         / (factorial(n + 1)))
                        * np.prod(np.array([new_x[index] - x[prod_index]
                                            for prod_index in range(n + 1)])
                                  ))]
                    for index in range(len(new_x))]]))


def main():
    func = cos(sympy_x)

    left = 1
    right = 6

    n_start, n_end = 3, 8

    for n in range(n_start, n_end):
        plt.subplot(2, 3, n - n_start + 1)
        interpolation(n=n, left=left, right=right, func=func, result_file=f'Lagrange^{n}.txt')

    plt.show()


if __name__ == '__main__':
    main()
