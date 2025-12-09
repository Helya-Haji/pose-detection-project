

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ==========================
# تنظیمات
# ==========================
input_video_path = r"D:\helyia\helya haji\kh.haji\yolo11-posedetection\day_6.avi"
output_video_path = r"D:\helyia\helya haji\kh.haji\yolo11-posedetection\day_6output.avi"
csv_path = r"C:\Users\User\Downloads\data_all.csv"
pose_model_path = "yolo11s-pose.pt"  # مدل YOLO pose

# ==========================
# بارگذاری مدل YOLO
# ==========================
pose_model = YOLO(pose_model_path)

# ==========================
# توابع کمکی
# ==========================
def angle(p1, p2, p3):
    """محاسبه زاویه بین سه نقطه"""
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

def compute_features_from_row(row):
    """
    row: یک ردیف از CSV
    خروجی: numpy array با 3 فیچر
    neck_angle, torso_angle, shoulder_diff_y
    """
    try:
        # Mapping از ستون‌های CSV موجود (head_x, head_y, JAB, JAC, ABC, BCE, JAD, KDE)
        head = np.array([row['head_x'], row['head_y']])
        # فرض می‌کنیم JAB=neck, ABC=r_shoulder, BCE=l_shoulder, KDE=mid_hip (مثال تقریبی)
        neck = np.array([row['JAB'], row['JAC']])
        r_shoulder = np.array([row['ABC'], row['BCE']])
        l_shoulder = np.array([row['JAD'], row['KDE']])
        mid_hip = np.array([row['NIH'], row['MGF']])

        neck_angle = angle(head, neck, mid_hip)
        shoulder_mid = (r_shoulder + l_shoulder) / 2
        torso_angle = angle(shoulder_mid, mid_hip, (mid_hip[0], mid_hip[1]-100))
        shoulder_diff_y = abs(r_shoulder[1] - l_shoulder[1])

        return np.array([neck_angle, torso_angle, shoulder_diff_y])
    except:
        return np.array([0,0,0])

# ==========================
# خواندن CSV و آماده‌سازی دیتاست
# ==========================
df = pd.read_csv(csv_path)

# ساخت label binary: فرض کنیم 'forward' = Bad (1), بقیه = Good (0)
df['label_binary'] = df['label'].apply(lambda x: 1 if x=='forward' else 0)

# محاسبه فیچرها
features_list = df.apply(compute_features_from_row, axis=1)
features = np.stack(features_list.values)
labels = df['label_binary'].values

# نرمال‌سازی فیچرها
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# تقسیم train/test
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# آموزش مدل RandomForest
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# ارزیابی مدل
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# ==========================
# پردازش ویدیوی RGB
# ==========================
cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO pose prediction
    results = pose_model(frame)
    keypoints_list = results[0].keypoints.xy if results else None

    if keypoints_list is not None and len(keypoints_list) > 0:
        keypoints = keypoints_list[0].cpu().numpy()  # فقط نفر اول
        try:
            # استخراج فیچر مشابه CSV
            head = keypoints[0]
            neck = keypoints[1]
            r_shoulder = keypoints[2]
            l_shoulder = keypoints[5]
            mid_hip = keypoints[8]

            neck_angle = angle(head, neck, mid_hip)
            shoulder_mid = (r_shoulder + l_shoulder)/2
            torso_angle = angle(shoulder_mid, mid_hip, (mid_hip[0], mid_hip[1]-100))
            shoulder_diff_y = abs(r_shoulder[1] - l_shoulder[1])

            features_frame = np.array([[neck_angle, torso_angle, shoulder_diff_y]])
            features_frame_scaled = scaler.transform(features_frame)
            pred = clf.predict(features_frame_scaled)[0]
            label_text = "Good Posture" if pred==0 else "Bad Posture"

            # رسم keypoints
            for kp in keypoints:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 5, (0,255,0), -1)

            # نوشتن label روی فریم
            cv2.putText(frame, label_text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0,0,255) if pred==1 else (0,255,0), 3)
        except:
            cv2.putText(frame, "Feature Error", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255),3)
    else:
        cv2.putText(frame, "No Person Detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255),3)

    out.write(frame)

cap.release()
out.release()
print(f"Output video saved to {output_video_path}")