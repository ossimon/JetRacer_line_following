import sys
import os

from optimizer import BayesianOptimizer
from robot import Robot

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from time import time

from fuzzy_rules import *
from membership_functions import *

FUZZY_RES = int(os.getenv("FUZZY_RES", "51"))  # was 101

class VectorizedFuzzyEvaluator:
    def __init__(self, membership_functions, rules, res=51):
        # universes
        self.U_speed = np.linspace(0, 1, res, dtype=np.float32)
        self.U_steer = np.linspace(-1, 1, res, dtype=np.float32)

        # store MF triangles for antecedents & consequents
        self.mf_points = membership_functions  # dict with 5 points each

        # Build consequent membership arrays (triangles) once
        self.speed_mfs = self._build_var_mfs(self.U_speed, 'speed')
        self.steer_mfs = self._build_var_mfs(self.U_steer, 'steer')

        # Encode rules as indices for fast lookup
        lv = linguistic_variables  # from your file
        self.idx = {
            'offset': {name:i for i, name in enumerate(lv['offset'])},
            'direction': {name:i for i, name in enumerate(lv['direction'])},
            'speed': {name:i for i, name in enumerate(lv['speed'])},
            'steer': {name:i for i, name in enumerate(lv['steer'])},
        }
        # Split speed/steer rules into compact lists
        self.speed_rules = []
        self.steer_rules = []
        for r in rules:
            (v1, l1), (v2, l2) = r['antecedents']
            cvar, clab = r['consequent']
            antecedent = (self.idx[v1][l1], self.idx[v2][l2])  # indices 0..4
            if cvar == 'speed':
                self.speed_rules.append((antecedent, self.idx['speed'][clab]))
            else:
                self.steer_rules.append((antecedent, self.idx['steer'][clab]))

    @staticmethod
    def _trimf(U, abc):
        a, b, c = abc
        y = np.zeros_like(U, dtype=np.float32)
        # rising
        mask = (a < U) & (U <= b)
        y[mask] = (U[mask] - a) / (b - a + 1e-12)
        # top
        y[U == b] = 1.0
        # falling
        mask = (b < U) & (U < c)
        y[mask] = (c - U[mask]) / (c - b + 1e-12)
        return y

    def _build_var_mfs(self, U, varname):
        pts = self.mf_points[varname]  # 5 points
        tris = []
        for i in range(5):
            left   = pts[max(0, i-1)]
            center = pts[i]
            right  = pts[min(4, i+1)]
            tris.append(self._trimf(U, (left, center, right)))
        return np.stack(tris, axis=0)  # [5, len(U)]

    @staticmethod
    def _interp_degree(x, pts):
        # Triangular grid with 5 centers; piecewise linear interp is cheap:
        # Find nearest two points and linear-interpolate (works because shapes are triangles)
        # For clarity, use skfuzzy once; but we can do direct math if needed.
        # Here we evaluate all 5 triangles at scalar x:
        degrees = np.zeros(5, dtype=np.float32)
        for i in range(5):
            a = pts[max(0, i-1)]
            b = pts[i]
            c = pts[min(4, i+1)]
            if a < x <= b:
                degrees[i] = (x - a) / (b - a + 1e-12)
            elif b < x < c:
                degrees[i] = (c - x) / (c - b + 1e-12)
            elif x == b:
                degrees[i] = 1.0
        return degrees

    def _aggregate_and_defuzz(self, rules, U, conseq_mfs, method='mom'):
        # rules: list of ((idx_offset, idx_direction), idx_conseq_label)
        # Get fuzzy degrees for current crisp inputs (stored during call)
        off_deg = self._cached_off_deg
        dir_deg = self._cached_dir_deg

        # Fire each rule: min(off_deg[i], dir_deg[j])
        # Aggregate: max over (rule_strength * triangle via clipping)
        agg = np.zeros_like(U, dtype=np.float32)
        for (i, j), k in rules:
            strength = off_deg[i] if off_deg[i] < dir_deg[j] else dir_deg[j]
            if strength <= 0.0:
                continue
            # clip the consequent MF at 'strength' and max-aggregate
            mf = conseq_mfs[k]
            # agg = np.maximum(agg, np.minimum(mf, strength))
            # inline for speed:
            clipped = np.where(mf < strength, mf, strength)
            agg = np.where(agg > clipped, agg, clipped)

        if method == 'som':  # smallest (leftmost) of maxima
            m = agg.max()
            idx = np.argmax(agg >= m - 1e-12)
            return float(U[idx])
        elif method == 'lom':  # largest (rightmost) of maxima
            m = agg.max()
            idx = len(U) - 1 - np.argmax((agg[::-1] >= m - 1e-12))
            return float(U[idx])
        else:  # 'mom' mean of maxima
            m = agg.max()
            if m <= 1e-12:
                return float(U[0])
            idxs = np.nonzero(np.abs(agg - m) <= 1e-6)[0]
            return float(U[idxs].mean())

    def compute(self, offset, direction):
        # cache antecedent degrees for this crisp input
        self._cached_off_deg = self._interp_degree(float(offset), self.mf_points['offset'])
        self._cached_dir_deg = self._interp_degree(float(direction), self.mf_points['direction'])

        speed = self._aggregate_and_defuzz(self.speed_rules, self.U_speed, self.speed_mfs, method='mom')
        steer = self._aggregate_and_defuzz(self.steer_rules, self.U_steer, self.steer_mfs, method='som')
        return speed, steer

class FuzzyAlgorithm:
    def __init__(self, fast=False):
        self.fast = fast
        self.fuzzy_system = None
        self.name = "Fuzzy"

    def set_genotype(self, genotype):
        mfs = mf_genotype_to_membership_functions(genotype[:10])


        rules = get_full_rules(
            speed_rules=genotype[10:23],
            steer_rules=genotype[23:36]
        )
        # fixed_rules = [3, 1, 4, 2, 4, 0, 2, 4, 2, 2, 2, 3, 4, 0, 1, 1, 1, 2, 0, 0, 1, 3, 4, 1, 1, 2]
        # rules = get_full_rules(
        #     speed_rules=fixed_rules[:13],
        #     steer_rules=fixed_rules[13:26]
        # )

        if self.fast:
            self.fuzzy_system = VectorizedFuzzyEvaluator(mfs, rules, res=FUZZY_RES)
        else:
            self.fuzzy_system = FuzzySystemBuilder(mfs, rules).build()

    def calculate_adjustments(self, observation):
        direction, offset, _ = observation
        # start = time()
        if self.fast:
            speed, steer = self.fuzzy_system.compute(offset, direction)
        else:
            self.fuzzy_system.input['direction'] = float(direction)
            self.fuzzy_system.input['offset'] = float(offset)
            self.fuzzy_system.compute()
            speed = float(self.fuzzy_system.output.get("speed", 0.0))
            steer = float(self.fuzzy_system.output.get("steer", 0.0))
        # end = time()
        # print(f"[Notification]: Fuzzy compute time: {(end - start)*1000:.2f} ms")

        return speed, -steer



class FuzzySystemBuilder:
    def __init__(self, membership_functions, rules, universes_of_discourse=None):
        # float32 + smaller grids
        off_U = np.linspace(-1,  1, FUZZY_RES, dtype=np.float32)
        dir_U = np.linspace(-1,  1, FUZZY_RES, dtype=np.float32)
        spd_U = np.linspace( 0,  1, FUZZY_RES, dtype=np.float32)
        str_U = np.linspace(-1,  1, FUZZY_RES, dtype=np.float32)

        self.offset = ctrl.Antecedent(off_U, 'offset')
        self.direction = ctrl.Antecedent(dir_U, 'direction')
        self.speed = ctrl.Consequent(spd_U, 'speed')
        self.steer = ctrl.Consequent(str_U, 'steer')

        # Cheaper defuzzifiers (centroid -> mom/som)
        self.speed.defuzzify_method = 'mom'   # mean of maxima
        self.steer.defuzzify_method = 'som'   # or 'lom' depending on your bias

        # Add membership functions (precast to float32)
        for var_name, values in linguistic_variables.items():
            mf_points = membership_functions[var_name]
            var = getattr(self, var_name)
            for idx, label in enumerate(values):
                left   = mf_points[max(0, idx - 1)]
                center = mf_points[idx]
                right  = mf_points[min(idx + 1, len(mf_points) - 1)]
                mf = fuzz.trimf(var.universe, [left, center, right]).astype(np.float32)
                var[label] = mf

        # Build rules (unchanged)
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

    x0 = [0.138660735238939, 0.005509326201965648, 0.965874633372737, 0.8908060430965608, 0.42972621981401515, 0.31726908875999077, 0.07264698209032718, 0.12101531617352726, 0.5757991366771309, 0.8159217712038822, 3, 1, 4, 2, 4, 0, 2, 4, 2, 2, 2, 3, 4, 0, 1, 1, 1, 2, 0, 0, 1, 3, 4, 1, 1, 2]
    x0 = None

    # Initialize fuzzy algorithm
    pid_algorithm = FuzzyAlgorithm(fast=True)
    genotype_bounds = [(float(lb), float(ub)) for lb, ub in mf_bounds] + \
        loosen_rule_bounds(speed_rule_bounds) + \
        loosen_rule_bounds(steer_rule_bounds)
    optimizer = BayesianOptimizer(pid_algorithm, robot, genotype_bounds, x0=x0, optimization_method='gbrt')

    try:
        best_genotype = optimizer.train(episodes=100)
        print(f"Best genotype found: {best_genotype}")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        robot.teardown()
        print("Robot resources cleaned up.")
