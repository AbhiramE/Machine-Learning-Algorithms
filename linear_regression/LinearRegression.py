from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import style

style.use('ggplot')


def best_fit_slope_and_intercept(x, y):
    m = (((mean(x) * mean(y)) - (mean(x * y))) /
         ((mean(x) ** 2) - mean(x ** 2)))

    b = mean(y) - m * mean(x)

    return m, b


def squarred_error(original, line):
    return sum((original - line) * (original - line))


def coefficient_of_determination(r_line, original_ys):
    ys_mean_line = [mean(original_ys) for y in original_ys]
    squared_error_regr = squarred_error(original_ys, r_line)
    squared_error_mean = squarred_error(original_ys, ys_mean_line)
    return 1 - (squared_error_regr / squared_error_mean)


def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

xs, ys = create_dataset(40, 40, 2, correlation='pos')

m, b = best_fit_slope_and_intercept(xs, ys)

regression_line = [(m * x) + b for x in xs]

predict_x = 7
predict_y = (m * predict_x) + b

r2 = coefficient_of_determination(regression_line, ys)
print(r2)

plt.scatter(xs, ys, color='#003F72', label='data')
plt.scatter(predict_x, predict_y, color="g")
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()
