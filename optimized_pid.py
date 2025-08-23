import sys
import os

from optimizer import BayesianOptimizer
from robot import Robot

P_DIRECTION_STEER = 0
I_DIRECTION_STEER = 1
D_DIRECTION_STEER = 2

P_OFFSET_STEER = 3
I_OFFSET_STEER = 4
D_OFFSET_STEER = 5

P_DIRECTION_SPEED = 6
I_DIRECTION_SPEED = 7
D_DIRECTION_SPEED = 8

P_OFFSET_SPEED = 9
I_OFFSET_SPEED = 10
D_OFFSET_SPEED = 11

BASE_SPEED = 12
MAX_SPEED = 13


class PidAlgorithm:
    def __init__(self):
        self.genotype = None
        self.prev_offset = 0
        self.prev_direction = 0
        self.name = "PID"

    def set_genotype(self, genotype):
        self.genotype = genotype
        self.prev_offset = 0
        self.prev_direction = 0

    def calculate_adjustments(self, observation):
        direction, offset, _ = observation
        speed_adjustment = self.calculate_speed(observation)
        steering_adjustment = self.calculate_steer(observation)

        self.prev_offset = offset
        self.prev_direction = direction

        return speed_adjustment, steering_adjustment

    def calculate_speed(self, observation):
        track_direction, track_offset, speed = observation
        offset = 1 - abs(track_offset)
        direction = 1 - abs(track_direction)

        offset_integral = offset + self.prev_offset
        offset_derivative = offset - self.prev_offset

        direction_integral = direction + self.prev_direction
        direction_derivative = direction - self.prev_direction

        max_speed_gain = self.genotype[P_DIRECTION_SPEED] + self.genotype[I_DIRECTION_SPEED] +\
            self.genotype[D_DIRECTION_SPEED] + self.genotype[P_OFFSET_SPEED] +\
            self.genotype[I_OFFSET_SPEED] + self.genotype[D_OFFSET_SPEED]

        if max_speed_gain == 0:
            return 0

        speed = self.genotype[P_DIRECTION_SPEED] * direction + \
            self.genotype[I_DIRECTION_SPEED] * direction_integral + \
            self.genotype[D_DIRECTION_SPEED] * direction_derivative + \
            self.genotype[P_OFFSET_SPEED] * offset + \
            self.genotype[I_OFFSET_SPEED] * offset_integral + \
            self.genotype[D_OFFSET_SPEED] * offset_derivative

        speed /= max_speed_gain
        speed = min(1, speed)
        speed = max(0, speed)

        return speed

    def calculate_steer(self, observation):
        direction, offset, speed = observation

        offset_integral = offset + self.prev_offset
        offset_derivative = offset - self.prev_offset

        direction_integral = direction + self.prev_direction
        direction_derivative = direction - self.prev_direction

        steer = self.genotype[P_DIRECTION_STEER] * direction + \
                self.genotype[I_DIRECTION_STEER] * direction_integral + \
                self.genotype[D_DIRECTION_STEER] * direction_derivative + \
                self.genotype[P_OFFSET_STEER] * offset + \
                self.genotype[I_OFFSET_STEER] * offset_integral + \
                self.genotype[D_OFFSET_STEER] * offset_derivative

        steer *= -1

        max_steer_gain = self.genotype[P_DIRECTION_STEER] + self.genotype[I_DIRECTION_STEER] + self.genotype[
            D_DIRECTION_STEER] + self.genotype[P_OFFSET_STEER] + self.genotype[I_OFFSET_STEER] + self.genotype[D_OFFSET_STEER]

        if max_steer_gain == 0:
            return 0

        # steer /= max_steer_gain
        steer = min(1, steer)
        steer = max(-1, steer)
        return steer


if __name__ == "__main__":
    robot = Robot()

    # Define genotype bounds
    genotype_bounds = {
        P_DIRECTION_STEER: (0., 1.),
        I_DIRECTION_STEER: (0., 1.),
        D_DIRECTION_STEER: (0., 1.),

        P_OFFSET_STEER: (0., 1.),
        I_OFFSET_STEER: (0., 1.),
        D_OFFSET_STEER: (0., 1.),

        P_DIRECTION_SPEED: (0., 1.),
        I_DIRECTION_SPEED: (0., 1.),
        D_DIRECTION_SPEED: (0., 1.),

        P_OFFSET_SPEED: (0., 1.),
        I_OFFSET_SPEED: (0., 1.),
        D_OFFSET_SPEED: (0., 1.),
    }
    genotype_bounds = [genotype_bounds[key] for key in sorted(genotype_bounds.keys())]

    x0 = [0.9229231752127518, 0.003104307638119864, 0.3939221095436195, 0.24812083376742416, 0.09756437302332513, 0.24176753896760922, 0.5513373507305395, 0.1781432853226739, 0.03405578276421895, 0.6237322830438287, 0.96674476411587, 0.40682901438372887]

    # Initialize PID algorithm
    pid_algorithm = PidAlgorithm()
    optimizer = BayesianOptimizer(pid_algorithm, robot, genotype_bounds, x0=x0)

    try:
        best_genotype = optimizer.train(episodes=37)
        print(f"Best genotype found: {best_genotype}")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        robot.teardown()
        print("Robot resources cleaned up.")
