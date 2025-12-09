

from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Settings
# -----------------------------
input_video_path = r"D:\helyia\helya haji\kh.haji\yolo11-posedetection\day_6_Trim.mp4"
output_video_path = r"D:\helyia\helya haji\kh.haji\yolo11-posedetection\day_6_pose_labeled.mp4"
csv_path = r"C:\Users\User\Downloads\data_all.csv"
pose_model_path = "yolo11s-pose.pt"

# -----------------------------
# Load dataset and train RandomForest
# -----------------------------
df = pd.read_csv(csv_path)
df['label_binary'] = df['label'].apply(lambda x: 1 if x=='forward' else 0)

def angle(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

def compute_features_from_row(row):
    try:
        head = np.array([row['head_x'], row['head_y']])
        neck = np.array([row['JAB'], row['JAC']])
        r_shoulder = np.array([row['ABC'], row['BCE']])
        l_shoulder = np.array([row['JAD'], row['KDE']])
        mid_hip = np.array([row['NIH'], row['MGF']])
        neck_angle = angle(head, neck, mid_hip)
        shoulder_mid = (r_shoulder + l_shoulder) / 2
        torso_angle = angle(shoulder_mid, mid_hip, (mid_hip[0], mid_hip[1]-100))
        shoulder_diff_y = abs(r_shoulder[1] - l_shoulder[1])
        return [neck_angle, torso_angle, shoulder_diff_y]
    except:
        return [0,0,0]

features_list = df.apply(compute_features_from_row, axis=1)
features = np.stack(features_list.values)
labels = df['label_binary'].values

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(features_scaled, labels)

# -----------------------------
# Load YOLO pose model
# -----------------------------
pose_model = YOLO(pose_model_path)

# -----------------------------
# Video capture and writer
# -----------------------------
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# -----------------------------
# Process video frame by frame (stream=True)
# -----------------------------
for result in pose_model.predict(source=input_video_path, stream=True, imgsz=640, conf=0.5):
    frame = result.orig_img.copy()

    # Iterate through all detected people
    for person_kps in result.keypoints.xy:
        kps = person_kps.cpu().numpy()
        if len(kps) < 9:
            continue

        # Draw keypoints
        for kp in kps:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Optional: draw skeleton (YOLO internal order)
        skeleton = [(0,1),(1,2),(1,5),(2,5),(2,8),(5,8)]
        for i,j in skeleton:
            pt1 = (int(kps[i][0]), int(kps[i][1]))
            pt2 = (int(kps[j][0]), int(kps[j][1]))
            cv2.line(frame, pt1, pt2, (255,0,0), 2)

        # Compute posture features
        head, neck = kps[0], kps[1]
        r_shoulder, l_shoulder = kps[2], kps[5]
        mid_hip = kps[8]

        neck_angle = angle(head, neck, mid_hip)
        shoulder_mid = (r_shoulder + l_shoulder)/2
        torso_angle = angle(shoulder_mid, mid_hip, (mid_hip[0], mid_hip[1]-100))
        shoulder_diff_y = abs(r_shoulder[1] - l_shoulder[1])

        features_frame = np.array([[neck_angle, torso_angle, shoulder_diff_y]])
        features_scaled_frame = scaler.transform(features_frame)
        pred = clf.predict(features_scaled_frame)[0]
        label_text = "Good Posture" if pred==0 else "Bad Posture"

        # Write label above head
        x_label, y_label = int(head[0]), int(head[1])-20
        cv2.putText(frame, label_text, (x_label, y_label),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0) if pred==0 else (0,0,255), 2)

    out.write(frame)

cap.release()
out.release()
print(f"Output video saved to {output_video_path}")

