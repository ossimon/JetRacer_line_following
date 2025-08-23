from time import time

import numpy as np
from skopt import gp_minimize, forest_minimize, gbrt_minimize

from robot import Robot


class BayesianOptimizer:
    def __init__(self, algorithm, robot, gen_bounds, x0=None, evaluation_steps=np.inf, optimization_method='gp'):
        self.algorithm = algorithm
        self.robot = robot
        self.genotype_bounds = gen_bounds
        self.x0 = x0
        self.evaluation_steps = evaluation_steps
        self.episode = 0
        optimization_methods = {
            'gp': gp_minimize,
            'forest': forest_minimize,
            'gbrt': gbrt_minimize
        }
        if optimization_method not in optimization_methods:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        self.optimizer = optimization_methods[optimization_method]

        self.best_genotype = None
        self.best_reward = float('-inf')

        self.last_timestamp = time()

    def evaluate_genotype(self, genotype):
        elapsed_time = time() - self.last_timestamp
        print(f"[Notification]: Time elapsed: {elapsed_time:.2f} seconds")

        print(f"Episode: {self.episode}")
        self.episode += 1

        self.algorithm.set_genotype(genotype)
        print(f"[Notification]: Evaluating genotype: {[round(g, 3) for g in genotype]}")
        
        total_reward, total_distance, offset_mse, total_time = self.robot.run(self.algorithm, self.episode)

        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_genotype = [round(g, 3) for g in genotype]
        
        print(f"[Notification]: Total reward: {total_reward:.2f}\n"
              f"[Notification]: Best genotype: {self.best_genotype}\n"
              f"[Notification]: Best reward: {self.best_reward:.2f}")
        
        self.last_timestamp = time()
        return -1 * total_reward

    def train(self, episodes):
        result = self.optimizer(
            self.evaluate_genotype,
            self.genotype_bounds,
            n_calls=episodes,
            random_state=0,
            x0=self.x0,
            n_jobs=1
        )

        print(f"Best Genotype: {result.x}")
        print(f"Best Reward: {result.fun}")
        return result.x
    
