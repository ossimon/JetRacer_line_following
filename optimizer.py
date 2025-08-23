from time import time

import numpy as np
from skopt import gp_minimize, forest_minimize, gbrt_minimize

from robot import Robot


class BayesianOptimizer:
    def __init__(self, algorithm, robot, gen_bounds, x0=None, evaluation_steps=np.inf, optimization_method='gp'):
        self.algorithm = algorithm
        self.robot = robot
        self.genotype_bounds = gen_bounds
        self.evaluation_steps = evaluation_steps
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

        # try to read existing results
        self.x0 = []
        self.y0 = []
        with open(f"{self.algorithm.name}_training_results.csv", "r") as f:
            for line in f:
                parts = line.strip().split(";")
                if len(parts) >= 3:
                    reward = float(parts[1])
                    # genotype = eval(parts[2])
                    # cannot use eval, python version issues
                    genotype = [float(x) for x in parts[2].strip("[]").split(", ")]
                    self.y0.append(-1 * reward)
                    self.x0.append(genotype)
        if len(self.x0) == 0:
            self.x0 = x0
            self.y0 = None
        else:
            print(f"[Notification]: Loaded {len(self.x0)} existing evaluations from {self.algorithm.name}_training_results.csv")
            # print(f"[Notification]: x0: {self.x0}")
            # print(f"[Notification]: y0: {self.y0}")

        self.episode = len(self.x0)

        self.last_timestamp = time()

    def evaluate_genotype(self, genotype):
        elapsed_time = time() - self.last_timestamp
        print(f"[Notification]: Time elapsed: {elapsed_time:.2f} seconds")

        print(f"Episode: {self.episode}")

        self.algorithm.set_genotype(genotype)
        print(f"[Notification]: Evaluating genotype: {[round(g, 3) for g in genotype]}")
        
        total_reward, total_distance, offset_mse, total_time, part_of_track_completed, avg_speed = self.robot.run(self.algorithm, self.episode)

        with open(f"{self.algorithm.name}_training_results.csv", "a") as f:
            f.write(f"{self.episode};{total_reward};{genotype};{total_time};{part_of_track_completed};{avg_speed};{offset_mse}\n")

        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_genotype = [round(g, 3) for g in genotype]
        
        print(f"[Notification]: Best genotype: {self.best_genotype}\n"
              f"[Notification]: Best reward: {self.best_reward:.2f}")
        
        self.episode += 1
        self.last_timestamp = time()
        return -1 * total_reward

    def train(self, episodes):
        self.robot.get_input()
        result = self.optimizer(
            self.evaluate_genotype,
            self.genotype_bounds,
            n_calls=episodes,
            random_state=0,
            x0=self.x0,
            y0=self.y0,
            n_jobs=1
        )

        print(f"Best Genotype: {result.x}")
        print(f"Best Reward: {result.fun}")
        return result.x
    
