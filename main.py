import face_recognition
import cv2
import numpy as np
import time
import mediapipe as mp

def draw_complete_face_mesh(frame, face_landmarks):
    mesh_color = (255, 255, 0)  # Cyan
    point_color = (255, 255, 255)  # Putih
    thickness = 1
    
    connections = [
        list(range(0, 17)),        # Kontur wajah
        list(range(17, 22)),       # Alis kiri
        list(range(22, 27)),       # Alis kanan
        list(range(27, 31)),       # Jembatan hidung
        list(range(31, 36)),       # Ujung hidung
        list(range(36, 42)) + [36],  # Mata kiri
        list(range(42, 48)) + [42],  # Mata kanan
        list(range(48, 60)) + [48],  # Bibir luar
        list(range(60, 68)) + [60],  # Bibir dalam
        [8, 27, 28, 29, 30, 33, 51, 62, 66, 57, 8],   # Garis horizontal
        [17, 36, 48, 54, 26, 45, 36],                 # Garis vertikal
        [27, 31, 32, 33, 34, 35, 30, 29, 28, 27],     # Detail hidung
        [17, 36], [18, 37], [19, 38], [20, 39], [21, 40], 
        [22, 41], [23, 42], [24, 43], [25, 44], [26, 45], # Alis ke mata
        [36, 27], [39, 28], [42, 27], [45, 28],           # Mata ke hidung
        [33, 51], [34, 52], [35, 53],                     # Hidung ke bibir
    ]
    
    for feature_points in face_landmarks.values():
        for point in feature_points:
            cv2.circle(frame, point, 2, point_color, -1)

    for connection in connections:
        for i in range(len(connection) - 1):
            try:
                all_points = []
                for points in face_landmarks.values():
                    all_points.extend(points)
                if connection[i] < len(all_points) and connection[i+1] < len(all_points):
                    pt1 = all_points[connection[i]]
                    pt2 = all_points[connection[i+1]]
                    cv2.line(frame, pt1, pt2, mesh_color, thickness)
            except:
                continue

def draw_hand_mesh(frame, hand_landmarks):
    hand_color = (0, 255, 0)
    connection_color = (255, 0, 0)
    thickness = 2
    
    hand_connections = mp.solutions.hands.HAND_CONNECTIONS
    
    h, w, c = frame.shape
    for landmark in hand_landmarks.landmark:
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (cx, cy), 3, hand_color, -1)
    
    for connection in hand_connections:
        start_idx, end_idx = connection
        start_point = hand_landmarks.landmark[start_idx]
        end_point = hand_landmarks.landmark[end_idx]
        start_x, start_y = int(start_point.x * w), int(start_point.y * h)
        end_x, end_y = int(end_point.x * w), int(end_point.y * h)
        cv2.line(frame, (start_x, start_y), (end_x, end_y), connection_color, thickness)

def draw_finger_details(frame, hand_landmarks):
    finger_tip_color = (0, 0, 255)
    finger_tips = [4, 8, 12, 16, 20]
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    
    h, w, c = frame.shape
    for i, tip_idx in enumerate(finger_tips):
        landmark = hand_landmarks.landmark[tip_idx]
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (cx, cy), 5, finger_tip_color, -1)
        cv2.putText(frame, finger_names[i], (cx + 10, cy), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, finger_tip_color, 1)

def draw_pose_mesh(frame, pose_landmarks):
    point_color = (0, 255, 255)
    connection_color = (255, 0, 255)
    thickness = 2
    
    h, w, c = frame.shape
    body_indices = [
        11, 12,  # bahu kiri, kanan
        13, 14,  # siku kiri, kanan
        15, 16,  # pergelangan tangan
        23, 24,  # pinggul kiri, kanan
        25, 26,  # lutut kiri, kanan
        27, 28,  # pergelangan kaki kiri, kanan
        29, 30,  # tumit kiri, kanan
        31, 32   # jari kaki kiri, kanan
    ]
    
    for idx in body_indices:
        landmark = pose_landmarks.landmark[idx]
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (cx, cy), 4, point_color, -1)

    pose_connections = [
        (11, 13), (13, 15),   # tangan kiri (bahu -> siku -> pergelangan)
        (12, 14), (14, 16),   # tangan kanan
        (11, 12),             # bahu kiri ke bahu kanan
        (11, 23), (12, 24),   # bahu ke pinggul
        (23, 24),             # pinggul kiri ke kanan
        (23, 25), (25, 27), (27, 29), (29, 31),  # kaki kiri
        (24, 26), (26, 28), (28, 30), (30, 32),  # kaki kanan
    ]
    
    for connection in pose_connections:
        start_idx, end_idx = connection
        start_point = pose_landmarks.landmark[start_idx]
        end_point = pose_landmarks.landmark[end_idx]
        start_x, start_y = int(start_point.x * w), int(start_point.y * h)
        end_x, end_y = int(end_point.x * w), int(end_point.y * h)
        cv2.line(frame, (start_x, start_y), (end_x, end_y), connection_color, thickness)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)

prev_time = 0
fps = 0

face_cache = []
CACHE_SIZE = 3

print("Deteksi mesh wajah, tangan, dan tubuh sedang berjalan...")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    current_time = time.time()
    if prev_time > 0:
        fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    display_frame = frame.copy()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    small_frame = cv2.resize(frame, (320, 240))
    rgb_small_frame = small_frame[:, :, ::-1]
    
    face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
    scale_factor_x = frame.shape[1] / small_frame.shape[1]
    scale_factor_y = frame.shape[0] / small_frame.shape[0]
    
    scaled_face_landmarks = []
    for landmarks in face_landmarks_list:
        scaled_landmarks = {}
        for feature, points in landmarks.items():
            scaled_landmarks[feature] = [
                (int(x * scale_factor_x), int(y * scale_factor_y)) 
                for (x, y) in points
            ]
        scaled_face_landmarks.append(scaled_landmarks)
    
    if scaled_face_landmarks:
        face_cache.append(scaled_face_landmarks)
        if len(face_cache) > CACHE_SIZE:
            face_cache.pop(0)
    
    # Deteksi tangan & tubuh
    hand_results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)
    
    if face_cache:
        current_landmarks = face_cache[-1]
        for landmarks in current_landmarks:
            draw_complete_face_mesh(display_frame, landmarks)
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            draw_hand_mesh(display_frame, hand_landmarks)
            draw_finger_details(display_frame, hand_landmarks)
    
    if pose_results.pose_landmarks:
        draw_pose_mesh(display_frame, pose_results.pose_landmarks)

    cv2.rectangle(display_frame, (8, 8), (250, 140), (0, 0, 0), -1)
    cv2.rectangle(display_frame, (8, 8), (250, 140), (255, 255, 255), 1)
    
    face_count = len(face_cache[-1]) if face_cache else 0
    hand_count = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0
    pose_count = 1 if pose_results.pose_landmarks else 0
    
    cv2.putText(display_frame, f"FPS: {int(fps)}", (15, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Faces: {face_count}", (15, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(display_frame, f"Hands: {hand_count}", (15, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(display_frame, f"Body: {pose_count}", (15, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    cv2.imshow('Face, Hand & Body Mesh Detection by erqyan', display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands.close()
pose.close()
video_capture.release()
cv2.destroyAllWindows()
