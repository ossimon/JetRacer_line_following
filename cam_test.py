import cv2
pipe=("nvarguscamerasrc sensor-id=0 ! "
      "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
      "nvvidconv ! video/x-raw, format=BGRx ! "
      "videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false")
cap=cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
print("Opened:", cap.isOpened())
ret, frame = cap.read()
print("Read:", ret, frame.shape if ret else None)
cap.release()
