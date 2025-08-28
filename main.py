from adafruit_servokit import ServoKit
# from datetime import datetime
import time
# import cv2
import atexit

i2c_address = 0x40
steering_channel = 0
throttle_channel = 1

class Robot:
    def __init__(self) -> None:
        self.steering_servo = None
        self.throttle_motor = None

        # ensure clean shutdown even on exceptions
        atexit.register(self._emergency_stop)

        try:
            self.setup()
            self.loop()
        finally:
            pass
            # self.teardown()

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
        # self.camera = CSICamera(width=1280, height=720, capture_fps=120)

        # OPTION B (uncomment to force ~30 fps): 1080p @ 30
        # self.camera = CSICamera(width=1920, height=1080, capture_fps=30)

        time.sleep(0.2)  # let first frame arrive
        # print("CSI camera initialized")

    def loop(self) -> None:
        instructions = [(0, 0), (90, 0.5), (90, -0.5), (180, 0), (90, 0)]

        for angle, throttle in instructions:
            self.steering_servo.angle = angle
            self.throttle_motor.throttle = throttle
            print(f"Set steering angle to {angle} and throttle to {throttle}")

            # frame = self.camera.read()
            # if frame is None:
            #     print("Warning: camera returned no frame")
            #     continue
            
            # h, w = frame.shape[:2]
            # ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            # print(f"[{ts}] Captured frame: {w}x{h}")

            # # Save as-is (BGR). No colorspace conversion needed.
            # cv2.imwrite(f"frames/frame_{ts}_angle{angle}_throttle{throttle}.jpg", frame)

            time.sleep(1)

        # Safety stop
        self.steering_servo.angle = 90
        self.throttle_motor.throttle = 0
        time.sleep(0.1)   # tiny delay before teardown

    # def teardown(self):
    #     try:
    #         if self.camera is not None and getattr(self.camera, "cap", None) is not None:
    #             self.camera.cap.release()
    #             time.sleep(0.1)  # let Argus unwind
    #     except Exception as e:
    #         print(f"Camera release warning: {e}")
    #     try:
    #         cv2.destroyAllWindows()
    #     except Exception as e:
    #         print(f"OpenCV cleanup warning: {e}")
    #     print("Cleaned up resources")

    def _emergency_stop(self):
        try:
            if self.steering_servo is not None:
                self.steering_servo.angle = 90
            if self.throttle_motor is not None:
                self.throttle_motor.throttle = 0
        except Exception:
            pass

if __name__ == "__main__":
    Robot()
