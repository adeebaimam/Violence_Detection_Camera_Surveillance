import cv2
import numpy as np
from collections import defaultdict
from playsound import playsound
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import yaml

# Load configuration
with open("config.yaml", "r") as f:
    raw_config = yaml.safe_load(f)
config = raw_config["profiles"][raw_config["default_profile"]]


# Load YOLOv8n-pose model
pose_model = YOLO(config['yolo_model_path'])

# Setup MediaPipe Hands
model_path = config['hand_model_path']
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=config['num_hands'],
)
hand_detector = vision.HandLandmarker.create_from_options(options)

def convert_to_mp_image(image):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def is_fist(hand_landmarks, width, height):
    """Detect if the hand is in a fist based on finger joint distances."""
    if not hand_landmarks:
        return False
    landmarks = [(lm.x * width, lm.y * height) for lm in hand_landmarks]  # Fixed syntax
    wrist = landmarks[0]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    # Check if fingers are curled (fist) by comparing tip-to-MCP distances
    finger_distances = [
        get_distance(index_tip, index_mcp),
        get_distance(middle_tip, middle_mcp)
    ]
    hand_size = get_distance(wrist, index_mcp)
    avg_finger_dist = np.mean(finger_distances)
    return avg_finger_dist < 0.4 * hand_size  # Stricter threshold for fist

def get_trajectory(history):
    """Calculate the direction vector of wrist movement."""
    if len(history) < 2:
        return None
    t1, p1 = history[-2]
    t2, p2 = history[-1]
    dt = t2 - t1 + 1e-6
    velocity = (np.array(p2) - np.array(p1)) / dt
    return velocity / (np.linalg.norm(velocity) + 1e-6)  # Normalized direction

cap = cv2.VideoCapture(config['input_video_path'])
cap.set(3, 640)
cap.set(4, 480)

start_time = cv2.getTickCount()
wrist_history = defaultdict(list)  # (person_id, wrist_key) -> [(time, pos), ...]
trigger_buffer = defaultdict(int)  # (personA_id, personB_id) -> frames of consistent action

while True:
    success, frame = cap.read()
    if not success:
        break

    height, width, _ = frame.shape
    people = []
    results = pose_model(frame)[0]
    keypoints, boxes = results.keypoints, results.boxes

    if keypoints is not None and boxes is not None:
        keypoints = keypoints.data.cpu().numpy()
        boxes = boxes.data.cpu().numpy()

        for i, (kps, box) in enumerate(zip(keypoints, boxes)):
            rwrist = tuple(kps[10][:2])
            lwrist = tuple(kps[9][:2])
            nose = tuple(kps[0][:2])
            neck = tuple((kps[5][:2] + kps[6][:2]) / 2)
            box_height = box[3] - box[1]  # For normalizing distances
            people.append({"id": i, "rwrist": rwrist, "lwrist": lwrist, "nose": nose, "neck": neck, "box": box, "box_height": box_height})

    mp_image = convert_to_mp_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    current_time = cv2.getTickCount()
    timestamp = int(((current_time - start_time) / cv2.getTickFrequency()) * 1e6)
    second_stamp = (current_time - start_time) / cv2.getTickFrequency()

    hand_result = hand_detector.detect_for_video(mp_image, timestamp)
    if hand_result.hand_landmarks:
        for hand_landmarks in hand_result.hand_landmarks:
            landmark_list = landmark_pb2.NormalizedLandmarkList(
                landmark=[landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks])
            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=landmark_list,
                connections=mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

    for i, personA in enumerate(people):
        for wrist_key in ["rwrist", "lwrist"]:
            wrist = personA[wrist_key]
            wrist_history[(personA["id"], wrist_key)].append((second_stamp, wrist))
            if len(wrist_history[(personA["id"], wrist_key)]) > 10:
                wrist_history[(personA["id"], wrist_key)].pop(0)

        for j, personB in enumerate(people):
            if i == j:
                continue

            iou = get_iou(personA["box"], personB["box"])
            if iou < 0.1:
                continue

            #if iou < 0.1 or iou > 0.6:
                # If people are too close (possibly standing idly), require motion
             #   near_but_idle = True
            #else:
             #   near_but_idle = False


            for wrist_key in ["rwrist", "lwrist"]:
                wrist = personA[wrist_key]
                neck = personB["neck"]
                nose = personB["nose"]
                wristB = personB["rwrist"]
                own_neck = personA["neck"]
                box_height = personA["box_height"]

                # Skip if wrist is closer to own neck (e.g., folded arms)
                if get_distance(wrist, own_neck) < get_distance(wrist, neck) * 0.5:
                    continue

                # Normalize distances by person height
                dist_neck = get_distance(wrist, neck) / box_height
                dist_face = get_distance(wrist, nose) / box_height
                dist_wrist = get_distance(wrist, wristB) / box_height

                history = wrist_history[(personA["id"], wrist_key)]
                condition_count = 0
                conditions = []

                # Condition 1: Wrist speed
                if len(history) >= 2:
                    t1, p1 = history[-2]
                    t2, p2 = history[-1]
                    v2 = get_distance(p1, p2) / (t2 - t1 + 1e-6)
                    conditions.append(v2 > config['wrist_speed_threshold'])
                    if conditions[-1]:
                        condition_count += 1

                # Condition 2: Wrist acceleration
                if len(history) >= 3:
                    t0, p0 = history[-3]
                    t1, p1 = history[-2]
                    t2, p2 = history[-1]
                    v1 = get_distance(p0, p1) / (t1 - t0 + 1e-6)
                    v2 = get_distance(p1, p2) / (t2 - t1 + 1e-6)
                    acc = (v2 - v1) / (t2 - t1 + 1e-6)
                    conditions.append(acc > config['acceleration_threshold'])
                    if conditions[-1]:
                        condition_count += 1
                else:
                    conditions.append(False)

                # Condition 3: Wrist near opponent's neck (normalized)
                conditions.append(dist_neck < config['distance_to_neck_thresh'] / box_height)
                if conditions[-1]:
                    condition_count += 1

                # Condition 4: Wrist near opponent's face (normalized)
                conditions.append(dist_face < config['distance_to_face_thresh'] / box_height)
                if conditions[-1]:
                    condition_count += 1

                # Condition 5: Fist detection
                is_fist_detected = False
                if hand_result.hand_landmarks:
                    for hand_landmarks in hand_result.hand_landmarks:
                        wrist_landmark = (hand_landmarks[0].x * width, hand_landmarks[0].y * height)
                        if get_distance(wrist_landmark, wrist) < config['fist_detection_radius']:
                            is_fist_detected = is_fist(hand_landmarks, width, height)
                            break
                conditions.append(is_fist_detected)
                if is_fist_detected:
                    condition_count += 1

                # Condition 6: Wrist trajectory toward opponent's face
                trajectory = get_trajectory(history)
                trajectory_condition = False
                if trajectory is not None:
                    face_vec = np.array(nose) - np.array(wrist)
                    face_vec = face_vec / (np.linalg.norm(face_vec) + 1e-6)
                    dot_product = np.dot(trajectory, face_vec)
                    trajectory_condition = dot_product > config['trajectory_dot_threshold']
                conditions.append(trajectory_condition)
                if trajectory_condition:
                    condition_count += 1
                


                # Require 4 conditions to increment fight score
                if condition_count >= config['min_trigger_conditions']:
                    trigger_buffer[(i, j)] += 1
                    # Debugging output to understand false positives
                    print(f"A{i}->B{j}: score={condition_count}, conditions={conditions}")
                else:
                    trigger_buffer[(i, j)] = 0

                if trigger_buffer[(i, j)] >= config['trigger_consistency_frames']:
                    cv2.putText(frame, f"FIGHT: A{i}->B{j}", (10, 30 + 20*i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    x1, y1, x2, y2 = personB["box"][:4].astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    playsound(config['alert_sound_path'], block=False)

    frame = results.plot()
    cv2.imshow("Fight Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

