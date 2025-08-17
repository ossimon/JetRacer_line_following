from jetcam.csi_camera import CSICamera
from adafruit_servokit import ServoKit
from datetime import datetime
import time
import cv2
import atexit
import time
import numpy as np
import os, sys

# import json library for configuration
import json

from track_recognition import extract_track, process_track_into_line, get_track_properties

i2c_address = 0x40
steering_channel = 0
throttle_channel = 1

MIN_SPEED = 0.05  # Minimum speed to avoid stopping completely
MAX_SPEED = 0.4  # Maximum speed for the robot

class Robot:
    def __init__(self, record_cam=False) -> None:
        self.record_cam = record_cam
        self.camera = None
        self.steering_servo = None
        self.throttle_motor = None
        self.speed = 0.0
        self.last_frame = None
        self.last_extracted_track = None
        self.recorded_frames = {}

        # ensure clean shutdown even on exceptions
        atexit.register(self.teardown)

        self.setup()

    def set_config(self, robot_config):
        self.robot_config = robot_config
        # print(f"Robot configuration updated: {self.robot_config}")

    def setup(self):
        try:
            kit = ServoKit(channels=16, address=0x40)
        except Exception as e:
            print(f"I2C/PCA9685 init failed: {e}")
            exit()
        print("ServoKit initialized")
        self.steering_servo = kit.servo[steering_channel]
        print("Steering servo initialized")
        self.throttle_motor = kit.continuous_servo[throttle_channel]
        print("Throttle motor initialized")

        # Camera (JetCam CSI)
        # OPTION A: stay at 720p → pick 60 fps (supported)
        self.camera = CSICamera(width=1280, height=720, capture_fps=120)

        # OPTION B (uncomment to force ~30 fps): 1080p @ 30
        # self.camera = CSICamera(width=1920, height=1080, capture_fps=30)

        time.sleep(0.2)  # let first frame arrive
        print("CSI camera initialized")

    def save_recorded_frames(self):
        if not self.recorded_frames:
            print("No frames recorded.")
            return

        os.makedirs("recorded_frames", exist_ok=True)
        for filename, frame in self.recorded_frames.items():
            print(f"Saving {filename}.png")
            if np.max(frame) <= 1:
                frame = (frame * 255).astype(np.uint8)
            cv2.imwrite(f"recorded_frames/{filename}.png", frame)
        print(f"Saved {len(self.recorded_frames)} recorded frames.")
        self.recorded_frames.clear()

    def get_observation(self):
        frame = self.camera.read()
        self.last_frame = frame  # Store the last frame for later use
        if frame is None:
            print("Warning: camera returned no frame")
            return None
        
        h, w = frame.shape[:2]
        ts = datetime.now().strftime("%H_%M_%S_%m_%d")
        # print(f"[{ts}] Captured frame: {w}x{h}")

        extracted_track = extract_track(frame, self.robot_config.get("binary_threshold", 60))
        self.last_extracted_track = extracted_track
        # print(np.unique(extracted_track, return_counts=True))

        if self.record_cam:
            self.recorded_frames[f"frame_{ts}"] = frame
            self.recorded_frames[f"extra_{ts}"] = extracted_track


        m, c = process_track_into_line(extracted_track)
        track_direction, track_offset = get_track_properties(m, c, extracted_track.shape)

        return track_direction, track_offset, self.speed
    
    def calculate_reward(self, observation):
        track_direction, track_offset, speed = observation
        max_offset = 1

        if track_offset == 0 and track_direction == 0:
            reward = -10
            # print(f"[Notification]: Robot is off-track. Offset: {track_offset:.2f}")
            # print(f"[Observation]: Reward: {reward:.2f}")
            return reward, True  # Large penalty for going off track and terminate episode

        reward = 0.1  # Base reward
        reward += (max_offset - abs(track_offset)) * 10  # deviation from line reward
        reward += speed  # Speed reward

        # if self.step_count % self.observation_log_interval == 0:
        # print(f"[Observation]: Reward: {reward:.2f}")

        return reward, False

    def set_controls(self, speed_adjustment, steering_adjustment):
        if not (-1 <= speed_adjustment <= 1):
            raise ValueError("Speed adjustment must be between -1 and 1")
        if not (-1 <= steering_adjustment <= 1):
            raise ValueError("Steering adjustment must be between -1 and 1")
        
        speed_adjustment = min(max(self.robot_config['min_speed'], speed_adjustment), self.robot_config['max_speed'])  # Ensure speed is non-negative
        steering_adjustment = max(-1, min(1, steering_adjustment))  # Ensure steering is within bounds

        for i in range(3):
            try:
                self.steering_servo.angle = (steering_adjustment + 1) * 90
                self.throttle_motor.throttle = speed_adjustment
                break  # Exit loop if successful
            except Exception as e:
                print(f"Error setting controls: {e}")
                time.sleep(0.1)
        # self.steering_servo.angle = (steering_adjustment + 1) * 90
        # self.throttle_motor.throttle = speed_adjustment

        self.speed = speed_adjustment
        # print(f"Set steering angle to {(steering_adjustment + 1) * 90} and throttle to {speed_adjustment}")

    def reset_robots_position_and_rotation(self):
        # Reset the robot's position and rotation logic here
        # This is a placeholder; actual implementation will depend on your environment setup
        print("Resetting robot's position and rotation to initial state.")
        self.steering_servo.angle = 90
        self.throttle_motor.throttle = 0

    def teardown(self):
        # Safety stop
        self.steering_servo.angle = 90
        self.throttle_motor.throttle = 0
        time.sleep(0.1)   # tiny delay before teardown
        try:
            if self.camera is not None and getattr(self.camera, "cap", None) is not None:
                self.camera.cap.release()
                time.sleep(0.1)  # let Argus unwind
        except Exception as e:
            print(f"Camera release warning: {e}")
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"OpenCV cleanup warning: {e}")
        print("Cleaned up resources")

    def _emergency_stop(self):
        try:
            if self.steering_servo is not None:
                self.steering_servo.angle = 90
            if self.throttle_motor is not None:
                self.throttle_motor.throttle = 0
        except Exception:
            pass


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


    def calculate_pid(self, observation):
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


    def train(self, episodes):
        for episode in range(episodes):
            # Load robot configuration from JSON file
            try:
                with open("robot_config.json", "r") as f:
                    robot_config = json.load(f)
                self.robot.set_config(robot_config)
            except FileNotFoundError:
                print("Warning: robot_config.json not found. Using default configuration.")
            except json.JSONDecodeError as e:
                print(f"Error parsing robot_config.json: {e}. Using default configuration.")

            print(f"[Notification]: Starting episode {episode + 1}.")
            usr_input = input("Send 's' to start.")
            while usr_input.lower() != 's':
                usr_input = input("Invalid input. Send 's' to start.")

            for _ in range(10):
                observation = self.robot.get_observation()
            if observation is None:
                print("[Error]: Failed to get observation. Skipping episode.")
                continue

            done = False
            prev_done = False
            total_reward = 0

            # self.robot.steering_servo.angle = 0
            # self.robot.throttle_motor.throttle = 1
            # input('type to stop')
            # robot.reset_robots_position_and_rotation()

            # last_signal_time = time.time()

            while not done or not prev_done:
                speed_adjustment, steering_adjustment = self.calculate_pid(observation)
                # observation, reward, done, _ = self.env.step(speed_adjustment, steering_adjustment)

                # while time.time() - last_signal_time < 0.05:  # 20 Hz control loop
                # # wait for next control loop
                #     time.sleep(0.01)
                robot.set_controls(speed_adjustment, steering_adjustment)
                # last_signal_time = time.time()

                observation = self.robot.get_observation()
                if observation is None:
                    print("[Error]: Failed to get observation. Ending episode.")
                    done = True
                    continue
                # total_reward += reward

                prev_done = done
                reward, done = self.robot.calculate_reward(observation)
                total_reward += reward

            print(f"[Notification]: Episode {episode + 1} ended. Total reward: {total_reward:.2f}")

            # Reset robot position and rotation for the next episode
            self.robot.reset_robots_position_and_rotation()
            if self.robot.record_cam:
                print(f"[Notification]: Saving recorded frames for episode {episode + 1}.")
                self.robot.save_recorded_frames()

            cv2.imwrite(f"debug_images/track_extracted_{episode + 1}.png", self.robot.last_frame)
            cv2.imwrite(f"debug_images/track_extracted_{episode + 1}_processed.png", self.robot.last_extracted_track)


if __name__ == "__main__":
    record_cam = len(sys.argv) > 1 and sys.argv[1] in ["1", "true", "yes", "y"]
    print(f"Recording camera: {record_cam}")
    # Define PID configuration
    config = {
        "kp": 2,
        "ki": 0.02,
        "kd": 0.25
    }


    # Robot
    robot = Robot(record_cam=record_cam)

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
