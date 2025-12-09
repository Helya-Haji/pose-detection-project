from ultralytics import YOLO

# Load pretrained YOLO11 pose model
model = YOLO("yolo11s-pose.pt")  # choose n/s/m/l/x depending on speed/accuracy

# Run inference on a video and save output
model.predict(
    source=r"D:\helyia\helya haji\kh.haji\yolo11-posedetection\day_6_testoutput_slow7.mp4",     # path to input video
    save=True,              # save annotated video
    project=r"D:\helyia\helya haji\kh.haji\yolo11-posedetection\runs\pose",    # folder where results go
    name="yolo11_pose",     # subfolder name
    imgsz=640,              # image size
    conf=0.5,               # confidence threshold
)
