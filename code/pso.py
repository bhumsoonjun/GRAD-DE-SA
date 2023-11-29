from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
import numpy as np
import benchmark_functions as bf
import matplotlib.pyplot as plt
import os

class MetricCollector:
    def __init__(self, name: str = None, run_id: int = None):
        self.best_val = -1
        self.name = name
        self.run_id = run_id
        self.iter_reached_vtr = -1
        self.nfev = -1
        self.time_to_vtr = -1
        self.total_time = -1
        self.time = -1
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
        path = f"results/pso/{self.name}/"

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

problems = [
    ("sphere", FunctionalProblem(30, sphere, xl=-100, xu=100), 10e-8),
    ("weighted_sphere", FunctionalProblem(30, weighted_sphere, xl=-100, xu=100), 10e-8),
    ("schwefel_1_2", FunctionalProblem(30, schwefel_1_2, xl=-500, xu=500), 10e-8),
    ("schwefel_2_3", FunctionalProblem(30, schwefel_2_3, xl=-500, xu=500), 1),
    ("easom", FunctionalProblem(2, easom, xl=-100, xu=100), 10e-8),
    ("rotated_hyper_ellipsoid", FunctionalProblem(30, rotated_hyper_ellipsoid, xl=-65.536, xu=65.536), 10e-8),
    ("rosenbrock", FunctionalProblem(30, rosenbrock, xl=-2.048, xu=2.048), 10e-8),
    ("griewangk", FunctionalProblem(30, griewangk, xl=-600, xu=600), 10e-8),
    ("pow_sum", FunctionalProblem(30, pow_sum, xl=-1, xu=1), 10e-8),
    ("ackley", FunctionalProblem(30, ackley, xl=-32.768, xu=32.768), 10e-8),
    ("rastrigin", FunctionalProblem(30, rastrigin, xl=-600, xu=600), 10e-8),
    ("dejong_5", FunctionalProblem(2, dejong_5, xl=-65.536, xu=65.536), 0.999),
    ("dejong_3", FunctionalProblem(30, dejong_3, xl=-3.5, xu=3.8), 10e-8),
]

for test_name, problem, vtr in problems:
    for i in range(1):
        print(f"Test: {test_name}, Run: {i}")
        metric = MetricCollector(test_name, i)
        metric = minimize(
            problem,
            PSO(
                pop_size=40,
                n_particles=100,
                vtr=vtr,
                metric=metric,
            ),
            termination=('n_iter', 1000),
            seed=30,
        )
        print(metric.best_val)
        print(len(metric.best_cost_each_gen))
        metric.save_results()