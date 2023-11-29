import numpy as np
import scipy.optimize as opt
import benchmark_functions as bf
import matplotlib.pyplot as plt
import os
import time
import sys
import scipy.stats as stats
class MetricCollector:
    def __init__(self, name: str = None, run_id: int = None):
        self.best_val = -1
        self.name = name
        self.run_id = run_id
        self.iter_reached_vtr = -1
        self.nfev = -1
        self.time_to_vtr = -1
        self.total_time = -1
        self.best_cost_each_gen = []

    def collect_vtr(self, i, nfev, time):
        self.iter_reached_vtr = i
        self.nfev = nfev
        self.time = time

    def collect_best_cost(self, best_cost):
        self.best_cost_each_gen.append(best_cost)

    def collect_last(self, last_val, total_time):
        self.best_val = last_val
        self.total_time = total_time

    def save_results(self):
        path = f"results/da/{self.name}/"

        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp = np.array(self.best_cost_each_gen)
        np.save(f"{path}cost.npy", self.best_cost_each_gen)
        # with open(path, "w") as f:
        #     f.write(f"{self.best_val}\n")
        #     f.write(f"{self.iter_reached_vtr}\n")
        #     f.write(f"{self.nfev}\n")
        #     f.write(f"{self.time}\n")
        #     f.write(f"{self.total_time}\n")

def sphere(x: np.ndarray):
    return np.sum(np.apply_along_axis(lambda y: (y)**2, 0, x))

def weighted_sphere(x: np.ndarray):
    cost = 0
    for i in range(x.shape[0]):
        cost += (i + 1) * (x[i]) ** 2
    return cost

def schwefel_1_2(x: np.ndarray):
    cost = 0
    for i in range(x.shape[0]):
        cost += np.sum(np.apply_along_axis(lambda y: y, 0, x[:i+1])) ** 2
    return cost

def schwefel_2_3(x: np.ndarray):
    cost = 418.9829 * x.shape[0] - np.sum(np.apply_along_axis(lambda y: y * np.sin(np.sqrt(np.abs(y))), 0, x))
    return cost

def easom(x: np.ndarray):
    return bf.Easom()(x) + 1

def rotated_hyper_ellipsoid(x: np.ndarray):
    # rotated hyper ellipsoid
    cost = 0
    for i in range(x.shape[0]):
        cost += np.sum(np.apply_along_axis(lambda y: (y)**2, 0, x[:i+1]))
    return cost

def rosenbrock(x: np.ndarray):
    # rosenbrock
    cost = 0
    for i in range(x.shape[0] - 1):
        cost += 100 * (x[i + 1] - x[i]**2)**2 + (x[i] - 1)**2
    return cost

def griewangk(x: np.ndarray):
    cost = 1/4000 * np.sum(np.apply_along_axis(lambda y: y**2, 0, x))
    prod = 1
    for i in range(x.shape[0]):
        prod *= np.cos(x[i] / np.sqrt(i + 1))
    return cost - prod + 1

def pow_sum(x: np.ndarray):
    cost = 0
    for i in range(x.shape[0]):
        cost += np.abs(x[i]) ** (i + 1)
    return cost

def ackley(x: np.ndarray):
    # ackley function
    cost = -20 * np.exp(-0.2 * np.sqrt(1/x.shape[0] * np.sum(np.apply_along_axis(lambda y: y**2, 0, x)))) - np.exp(1/x.shape[0] * np.sum(np.apply_along_axis(lambda y: np.cos(2 * np.pi * y), 0, x))) + 20 + np.exp(1)
    return cost

def rastrigin(x: np.ndarray):
    return 10 * x.shape[0] + np.sum(np.apply_along_axis(lambda y: y**2 - 10 * np.cos(2 * np.pi * y), 0, x))

def dejong_5(x: np.ndarray):
    return bf.DeJong5()(x)

def dejong_3(x: np.ndarray):
    return bf.DeJong3(n_dimensions=x.shape[0])(x) + 120

test_funcs = [
    ("sphere", sphere, [(-100, 100) for _ in range(30)], 10e-8),
    ("weighted_sphere", weighted_sphere, [(-100, 100) for _ in range(30)], 10e-8),
    ("schwefel_1_2", schwefel_1_2, [(-500, 500) for _ in range(30)], 10e-8),
    ("schwefel_2_3", schwefel_2_3, [(-500, 500) for _ in range(30)], 1),
    ("easom", easom, [(-100, 100) for _ in range(2)], 10e-8),
    ("rotated_hyper_ellipsoid", rotated_hyper_ellipsoid, [(-65.536, 65.536) for _ in range(30)], 10e-8),
    ("rosenbrock", rosenbrock, [(-2.048, 2.048) for _ in range(30)], 10e-8),
    ("griewangk", griewangk, [(-600, 600) for _ in range(30)], 10e-8),
    ("pow_sum", pow_sum, [(-1, 1) for _ in range(30)], 10e-8),
    ("ackley", ackley, [(-32.768, 32.768) for _ in range(30)], 10e-8),
    ("rastrigin", rastrigin, [(-600, 600) for _ in range(30)], 10e-8),
    ("dejong_5", dejong_5, [(-65.536, 65.636) for _ in range(2)], 0.999),
    ("dejong_3", dejong_3, [(-3.5, 3.8)  for _ in range(30)], 10e-8),
]

for test_name, test_func, test_bounds, vtr in test_funcs:
    for i in range(1):
        print(f"Test: {test_name}, Run: {i}")
        metric = MetricCollector(name=test_name, run_id=i)
        opt.dual_annealing(
            test_func,
            test_bounds,
            maxiter=1000,
            metric=metric,
            vtr=vtr,
            seed=30,
        )
        metric.save_results()