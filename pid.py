import time
import time
import numpy as np
import os, sys

from robot import Robot


class PidController:
    def __init__(self, robot, config):
        self.robot = robot
        self.config = config
        self.kp = config.get("kp", 0.5)
        self.ki = config.get("ki", 0.01)
        self.kd = config.get("kd", 0.1)
        self.integral = 0
        self.prev_error = 0
        self.dt = 0
        self.last_timestamp = time.time()


    def _calculate_pid(self, observation):
        track_direction, track_offset, speed = observation

        current_time = time.time()
        self.dt = current_time - self.last_timestamp
        self.last_timestamp = current_time
        dt = max(self.dt, 1e-3)          # set self.dt once (e.g., 0.05)

        # blended error: lateral + heading
        error = 0.7  * (-track_offset) + 0.3 * (-track_direction)

        # derivative with simple low-pass filter
        raw_d = (error - self.prev_error) / dt
        self.d_filt = 0.8*self.d_filt + 0.2*raw_d if hasattr(self, "d_filt") else raw_d

        # gain scheduling with speed (optional, keeps behavior similar across speeds)
        # sp = max(speed, 1e-3)
        sp = 0
        kp = self.kp / (1 + 0.4*sp)
        kd = self.kd * (1 + 0.6*sp)
        ki = self.ki                    # usually keep small and constant

        # tentative PID
        u = kp*error + ki*self.integral + kd*self.d_filt

        # anti-windup: only integrate if not saturating against the error
        u_sat = max(min(u, 1.0), -1.0)
        if (u == u_sat) or (abs(u - u_sat) < 1e-6):
            if (u_sat*error) < 0:  # output fights the error → allow integration
                self.integral += error * dt
        else:
            self.integral += error * dt

        # recompute with updated integral
        u = kp*error + ki*self.integral + kd*self.d_filt
        steering = max(min(u, 1.0), -1.0)
        self.prev_error = error

        # throttle strategy: slow down for sharp turns
        base = 1
        throttle = max(0.2, base - 0.35*abs(steering))
        return throttle, steering
    
    def calculate_adjustments(self, observation):
        return self._calculate_pid(observation)

    def train(self, episodes):
        for episode in range(episodes):
            self.robot.run(self, episode)


if __name__ == "__main__":
    # Define PID configuration
    config = {
        "kp": 2,
        "ki": 0.02,
        "kd": 0.25
    }

    # Robot
    robot = Robot()

    # Initialize PID controller and train
    controller = PidController(robot, config)

    try:
        controller.train(episodes=50)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        robot.teardown()
        print("Robot resources cleaned up.")
