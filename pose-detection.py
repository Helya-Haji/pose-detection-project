import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

input_video_path = r"D:\6019-187893788.mp4"
output_video_path = r"\6019-187893788-output2.mp4"
csv_path = r"C:\Users\User\Downloads\data_all.csv"
pose_model_path = r"\yolo11-posedetection\yolo11s-pose.pt"

conf_threshold = 0.5

pose_model = YOLO(pose_model_path)


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
        torso_angle = angle(shoulder_mid, mid_hip, (mid_hip[0], mid_hip[1] - 100))
        shoulder_diff_y = abs(r_shoulder[1] - l_shoulder[1])

        return np.array([neck_angle, torso_angle, shoulder_diff_y])
    except:
        return np.array([0, 0, 0])

df = pd.read_csv(csv_path)
df['label_binary'] = df['label'].apply(lambda x: 1 if x == 'forward' else 0)

features_list = df.apply(compute_features_from_row, axis=1)
features = np.stack(features_list.values)
labels = df['label_binary'].values

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))


cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0
person_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    results = pose_model(frame, conf=conf_threshold, verbose=False)
    
    annotated_frame = results[0].plot() 
    
    if results[0].keypoints is not None and len(results[0].keypoints) > 0:
        for i, (box, kps) in enumerate(zip(results[0].boxes, results[0].keypoints)):
            try:
                kps_np = kps.xy.cpu().numpy()
                
                if kps_np.ndim == 3:
                    kps_np = kps_np.squeeze(0)
                
                if len(kps_np) >= 9:
                    head = kps_np[0]
                    neck = kps_np[1]
                    r_shoulder = kps_np[2]
                    l_shoulder = kps_np[5]
                    mid_hip = kps_np[8]
                    
                    head = [float(head[0]), float(head[1])]
                    neck = [float(neck[0]), float(neck[1])]
                    r_shoulder = [float(r_shoulder[0]), float(r_shoulder[1])]
                    l_shoulder = [float(l_shoulder[0]), float(l_shoulder[1])]
                    mid_hip = [float(mid_hip[0]), float(mid_hip[1])]

                    neck_angle = angle(head, neck, mid_hip)
                    shoulder_mid = [(r_shoulder[0] + l_shoulder[0]) / 2, 
                                   (r_shoulder[1] + l_shoulder[1]) / 2]
                    torso_angle = angle(shoulder_mid, mid_hip, (mid_hip[0], mid_hip[1] - 100))
                    shoulder_diff_y = abs(r_shoulder[1] - l_shoulder[1])
                    
                    feat = np.array([[neck_angle, torso_angle, shoulder_diff_y]])
                    feat_scaled = scaler.transform(feat)
                    pred = clf.predict(feat_scaled)[0]
                    
                    label_text = "Good Posture" if pred == 0 else "Bad Posture"
                    color_lbl = (0, 255, 0) if pred == 0 else (0, 0, 255)
                    
                    if box.xyxy is not None:
                        box_coords = box.xyxy[0].cpu().numpy()
                        x1, y1 = int(box_coords[0]), int(box_coords[1])
                        
                        font_scale = 0.8
                        thickness = 2
                        
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                        )
                        
                        cv2.rectangle(annotated_frame,
                                     (x1, y1 - text_height - 10),
                                     (x1 + text_width, y1),
                                     (0, 0, 0), -1)
                        
                        cv2.putText(annotated_frame, label_text,
                                   (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   font_scale,
                                   color_lbl,
                                   thickness)
                    
                    person_count += 1
                    
            except Exception as e:
                print(f"Error processing person {i+1} in frame {frame_count}: {e}")
                continue
    

    out.write(annotated_frame)
    
    if frame_count % 30 == 0:
        print(f"Processed frame {frame_count}...")

cap.release()
out.release()

print(f"\n compleate")
print(f" output saved at: {output_video_path}")
