import numpy as np
import matplotlib.pyplot as plt
from bokeh.models import ranges

# y = wx + b
# loss = Σ(wx + b - y)^2 >=0
def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))  # average

# 梯度信息
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]

def gradient_descent_runner(points, starting_b, starting_w, learning_rate, iterations):
    b = starting_b
    w = starting_w
    for i in range(iterations):
        b, w = step_gradient(b, w, points, learning_rate)
    return [b, w]

def run():
    points = np.genfromtxt("data.csv", delimiter=",") # (100,2)
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_w = 0 # initial slope guess
    iterations = 1000
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_error_for_line_given_points(initial_b, initial_w, points))
          )
    print("Running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".
          format(iterations, b, w,
                 compute_error_for_line_given_points(b, w, points))
          )

if __name__ == '__main__':
    run()