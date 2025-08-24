import sys
import os

from optimizer import BayesianOptimizer
from robot import Robot

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from fuzzy_rules import *
from membership_functions import *

class FuzzyAlgorithm:
    def __init__(self):
        self.fuzzy_system = None
        self.name = "Fuzzy"

    def set_genotype(self, genotype):
        mfs = mf_genotype_to_membership_functions(genotype[:10])
        rules = get_full_rules(
            speed_rules=genotype[10:23],
            steer_rules=genotype[23:36]
        )
        self.fuzzy_system = FuzzySystemBuilder(membership_functions=mfs, rules=rules).build()

    def calculate_adjustments(self, observation):
        direction, offset, _ = observation
        self.fuzzy_system.input['direction'] = direction
        self.fuzzy_system.input['offset'] = offset
        self.fuzzy_system.compute()
        speed_adjustment = self.fuzzy_system.output.get("speed", 0)
        steering_adjustment = self.fuzzy_system.output.get("steer", 0)
        return speed_adjustment, -steering_adjustment

class FuzzySystemBuilder:
    def __init__(self, membership_functions, rules, universes_of_discourse=None):
        # Define fuzzy variables
        self.offset = ctrl.Antecedent(np.linspace(-1, 1, 101), 'offset')
        self.direction = ctrl.Antecedent(np.linspace(-1, 1, 101), 'direction')
        self.speed = ctrl.Consequent(np.linspace(0, 1, 101), 'speed')
        self.steer = ctrl.Consequent(np.linspace(-1, 1, 101), 'steer')

        # Add membership functions
        for var_name, values in linguistic_variables.items():
            mf_points = membership_functions[var_name]
            var = getattr(self, var_name)
            for idx, label in enumerate(values):
                left = mf_points[max(0, idx - 1)]
                center = mf_points[idx]
                right = mf_points[min(idx + 1, len(mf_points) - 1)]
                var[label] = fuzz.trimf(var.universe, [left, center, right])

        # Build rules
        self.rule_list = []
        for rule in rules:
            antecedents = []
            for ant_var, ant_label in rule['antecedents']:
                antecedents.append(getattr(self, ant_var)[ant_label])
            consequent_var, consequent_label = rule['consequent']
            consequent = getattr(self, consequent_var)[consequent_label]
            self.rule_list.append(ctrl.Rule(antecedents[0] & antecedents[1], consequent))

        self.system = ctrl.ControlSystem(self.rule_list)

    def build(self):
        return ctrl.ControlSystemSimulation(self.system)

if __name__ == "__main__":
    robot = Robot()

    # x0 = [0.9229231752127518, 0.003104307638119864, 0.3939221095436195, 0.24812083376742416, 0.09756437302332513, 0.24176753896760922, 0.5513373507305395, 0.1781432853226739, 0.03405578276421895, 0.6237322830438287, 0.96674476411587, 0.40682901438372887]
    x0 = None

    # Initialize fuzzy algorithm
    pid_algorithm = FuzzyAlgorithm()
    genotype_bounds = [(float(lb), float(ub)) for lb, ub in mf_bounds] + \
        loosen_rule_bounds(speed_rule_bounds) + \
        loosen_rule_bounds(steer_rule_bounds)
    optimizer = BayesianOptimizer(pid_algorithm, robot, genotype_bounds, x0=x0)

    try:
        best_genotype = optimizer.train(episodes=11)
        print(f"Best genotype found: {best_genotype}")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        robot.teardown()
        print("Robot resources cleaned up.")
