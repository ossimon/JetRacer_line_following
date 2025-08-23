from jetcam.csi_camera import CSICamera
from adafruit_servokit import ServoKit
from datetime import datetime
import time
import cv2
import atexit
import time
import numpy as np
import os, sys
import json
import shutil
import keyboard

from track_recognition import extract_track, process_track_into_line, get_track_properties

i2c_address = 0x40
steering_channel = 0
throttle_channel = 1

MIN_SPEED = 0.05  # Minimum speed to avoid stopping completely
MAX_SPEED = 0.4  # Maximum speed for the robot

class Robot:
    def __init__(self) -> None:
        self.camera = None
        self.steering_servo = None
        self.throttle_motor = None
        self.speed = 0.0
        self.last_frame = None
        self.last_extracted_track = None
        self.recorded_frames = []
        self.extracted_tracks = []
        self.config = None
        self.last_distance_measurement_timestamp = time.time()
        self.distance_from_last_frame = 0.0

        # ensure clean shutdown even on exceptions
        atexit.register(self.teardown)

        self.setup()

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

        self.camera = CSICamera(width=1280, height=720, capture_fps=120)

        time.sleep(0.2)  # let first frame arrive
        print("CSI camera initialized")

    def save_recorded_frames(self):
        if len(self.recorded_frames) == 0:
            print("No frames recorded.")
            return

        shutil.rmtree("recorded_frames", ignore_errors=True)
        os.makedirs("recorded_frames", exist_ok=True)

        for frame, ts in self.recorded_frames:
            filename = f"frame_{ts}.png"
            cv2.imwrite(f"recorded_frames/{filename}.png", frame)
        for track, ts in self.extracted_tracks:
            if np.max(track) <= 1:
                track = (track * 255).astype(np.uint8)
            filename = f"track_{ts}.png"
            cv2.imwrite(f"recorded_frames/{filename}.png", track)
        # for filename, frame in self.recorded_frames.items():
        #     print(f"Saving {filename}.png")
        #     if np.max(frame) <= 1:
        #         frame = (frame * 255).astype(np.uint8)
        #     cv2.imwrite(f"recorded_frames/{filename}.png", frame)
        print(f"Saved {len(self.recorded_frames) + len(self.extracted_tracks)} recorded frames.")
        self.recorded_frames = []
        self.extracted_tracks = []

    def get_observation(self):
        frame = self.camera.read()
        self.last_frame = frame  # Store the last frame for later use
        if frame is None:
            print("Warning: camera returned no frame")
            return None
        
        h, w = frame.shape[:2]
        ts = datetime.now().strftime("%H_%M_%S_%m_%d")
        # print(f"[{ts}] Captured frame: {w}x{h}")

        extracted_track = extract_track(frame, self.config.get("binary_threshold", 60))
        self.last_extracted_track = extracted_track
        # print(np.unique(extracted_track, return_counts=True))

        if self.config["record_cam"] and (len(self.recorded_frames) == 0 or self.recorded_frames[-1][1] != ts):
            self.recorded_frames.append((frame, ts))
            self.extracted_tracks.append((extracted_track, ts))


        m, c = process_track_into_line(extracted_track)
        track_direction, track_offset = get_track_properties(m, c, extracted_track.shape)

        return track_direction, track_offset, self.speed
    
    def is_on_track(self, observation):
        track_direction, track_offset, _ = observation
        if track_offset == 0 and track_direction == 0:
            return False
        return True
    
    def set_controls_without_validation(self, speed_adjustment, steering_adjustment):
        self.steering_servo.angle = (steering_adjustment + 1) * 90
        self.throttle_motor.throttle = speed_adjustment
        self.speed = speed_adjustment
        # print(f"Set steering angle to {(steering_adjustment + 1) * 90} and throttle to {speed_adjustment}")

    def set_controls(self, speed_adjustment, steering_adjustment):
        if not (-1 <= speed_adjustment <= 1):
            raise ValueError("Speed adjustment must be between -1 and 1")
        if not (-1 <= steering_adjustment <= 1):
            raise ValueError("Steering adjustment must be between -1 and 1")
        
        current_time = time.time()
        self.distance_from_last_frame = self.speed * (current_time - self.last_distance_measurement_timestamp)
        self.total_distance += self.distance_from_last_frame
        self.last_distance_measurement_timestamp = current_time

        speed_adjustment = min(max(self.config['min_speed'], speed_adjustment), self.config['max_speed'])  # Ensure speed is non-negative
        steering_adjustment = max(-1, min(1, steering_adjustment))  # Ensure steering is within bounds

        attempts = 0
        for i in range(10):
            attempts = i + 1
            try:
                self.steering_servo.angle = (steering_adjustment + 1) * 90
                self.throttle_motor.throttle = speed_adjustment
                break  # Exit loop if successful
            except Exception as e:
                print(f"Error setting controls: {e}")
                time.sleep(0.1)
        if attempts == 10:
            print(f"[Error]: Failed to set controls after {attempts} attempts. Using last known values.")

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

    def get_input(self):
        print(
            "W - forward\n"
            "A - left\n"
            "D - right\n"
            "S - stop\n"
            "G - start episode\n"
        )
        moving_speed = 0.5
        steering_angle = 1
        while True:
            speed = 0.0
            steering = 0.0
            if keyboard.is_pressed('w'):
                speed += moving_speed
            if keyboard.is_pressed('s'):
                speed -= moving_speed
            if keyboard.is_pressed('a'):
                steering += steering_angle
            if keyboard.is_pressed('d'):
                steering -= steering_angle
            if keyboard.is_pressed('g'):
                break
            self.set_controls_without_validation(speed, steering)

            time.sleep(0.1)

    def run(self, algorithm, episode):
        self.get_input()

        # Load robot configuration from JSON file
        try:
            with open("robot_config.json", "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print("Warning: config.json not found. Using default configuration.")
        except json.JSONDecodeError as e:
            print(f"Error parsing config.json: {e}. Using default configuration.")


        print(f"[Notification]: Starting episode {episode + 1}.")

        # usr_input = input("Send 's' to start.")
        # while usr_input.lower() != 's':
        #     usr_input = input("Invalid input. Send 's' to start.")

        for _ in range(10):
            observation = self.get_observation()
        if observation is None:
            print("[Error]: Failed to get observation. Skipping episode.")
            return

        is_on_track = True
        total_reward = 0
        self.last_distance_measurement_timestamp = time.time()
        self.distance_from_last_frame = 0.0
        self.offsets = []

        self.start = time.time()
        self.total_distance = 0.0

        while is_on_track:
            speed_adjustment, steering_adjustment = algorithm.calculate_adjustments(observation)
            self.set_controls(speed_adjustment, steering_adjustment)

            observation = self.get_observation()
            if observation is None:
                print("[Error]: Failed to get observation. Ending episode.")
                break
            track_direction, track_offset, speed = observation
            self.offsets.append(track_offset)
            is_on_track = self.is_on_track(observation)
            
        total_time = time.time() - self.start
        offset_mse = np.mean(np.square(self.offsets))
        total_reward = self.total_distance * (1 / offset_mse) if offset_mse > 0 else 0

        print(f"[Notification]: Episode {episode + 1} ended.\n"
              f"Total time: {total_time:.2f} seconds\n"
              f"Total distance: {self.total_distance:.2f} meters**\n"
              f"Average offset MSE: {offset_mse:.4f}\n"
              f"Total reward: {total_reward:.2f}"
        )

        # Reset robot position and rotation for the next episode
        self.reset_robots_position_and_rotation()
        if self.config["record_cam"]:
            print(f"[Notification]: Saving recorded frames for episode {episode + 1}.")
            self.save_recorded_frames()

        cv2.imwrite(f"debug_images/frame.png", self.last_frame)
        cv2.imwrite(f"debug_images/track.png", self.last_extracted_track)

        return total_reward, self.total_distance, offset_mse, total_time
