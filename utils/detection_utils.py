import cv2
import dlib
import numpy as np
from imutils import face_utils
from collections import defaultdict
from ultralytics import YOLO
from scipy.spatial import distance as dist
from utils.logging_utils import log_detection
import mediapipe as mp


# ==============================
# Constants
# ==============================
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
HEAD_TURN_THRESHOLD = 20
MULTIPLE_FACE_THRESHOLD = 4
PROXIMITY_THRESHOLD = 100
LIP_MOVEMENT_THRESHOLD = 0.05
LIP_MOVEMENT_CONSEC_FRAMES = 5

# ==============================
#  load models
# ==============================
detector = None
predictor = None
yolo_model = None

# ==============================
# States
# ==============================
eye_counter = 0
prev_gaze = "center"
lip_movement_history = []
prev_lip_distance = 0
talk_counter = 0
missed_frames = defaultdict(int)


# ==============================
# Load models
# ==============================
def load_models():
    global detector, predictor
    if detector is None or predictor is None:
        print("[INFO] Loading dlib models...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        print("[INFO] Models loaded successfully.")


def load_yolo_model():
    global yolo_model
    if yolo_model is None:
        print("[INFO] Loading YOLO model for object detection...")
        yolo_model = YOLO("yolov8n.pt")  # Using nano version for speed
        print("[INFO] YOLO model loaded successfully.")

# ==============================
# Prohibited object detection
# ==============================
def detect_prohibited_objects(frame):
    global yolo_model
    load_yolo_model()

    prohibited_classes = [67]  # 67 = Cell phone
    detected_objects = []

    results = yolo_model(frame, verbose=False)
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id in prohibited_classes:
                    confidence = float(box.conf[0])
                    if confidence > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detected_objects.append({
                            "class": class_id,
                            "confidence": confidence,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                        })
    return detected_objects


# ==============================
# Lip movement (talking detection)
# ==============================
def detect_lip_movement(shape, frame):
    global lip_movement_history, prev_lip_distance, talk_counter
    lips = shape[48:68]

    A = np.linalg.norm(lips[6] - lips[0])  # Corner to corner
    B1 = np.linalg.norm(lips[2] - lips[10])  # Top to bottom
    B2 = np.linalg.norm(lips[4] - lips[8])
    B = (B1 + B2) / 2.0

    if A == 0:
        return False

    mar = B / A
    lip_movement_history.append(mar)
    if len(lip_movement_history) > 10:
        lip_movement_history.pop(0)

    if len(lip_movement_history) > 1:
        movement = abs(lip_movement_history[-1] - lip_movement_history[-2])
        if movement > LIP_MOVEMENT_THRESHOLD:
            talk_counter += 1
            if talk_counter >= LIP_MOVEMENT_CONSEC_FRAMES:
                talk_counter = 0
                return True
        else:
            talk_counter = max(0, talk_counter - 1)
    return False


# ==============================
# Face-based detections
# ==============================
def detect_multiple_faces(faces, frame):
    return len(faces) > MULTIPLE_FACE_THRESHOLD


def detect_person_proximity(faces, gray):
    if len(faces) < 2:
        return False

    centroids = []
    for face in faces:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        centroids.append((x + w / 2, y + h / 2))

    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            d = dist.euclidean(centroids[i], centroids[j])
            if d < PROXIMITY_THRESHOLD:
                return True
    return False


# ==============================
# Head and eye movement
# ==============================
def detect_head_turn(shape, frame):
    image_points = np.array([
        shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    size = frame.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    yaw = angles[1]

    if yaw < -HEAD_TURN_THRESHOLD:
        return True, "left"
    elif yaw > HEAD_TURN_THRESHOLD:
        return True, "right"
    else:
        return False, "center"


# ==============================
# Eye Movement Detection (Mediapipe)
# ==============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,  # IMPORTANT: gives iris landmarks
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)


def get_gaze_direction(shape, frame=None):
    """
    Uses Mediapipe iris landmarks to determine left/right gaze.
    Ignores up/down movement.
    """
    global face_mesh

    # Convert frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        return "center"

    face_landmarks = results.multi_face_landmarks[0]

    # Mediapipe iris landmarks (normalized coords)
    # Left iris = 468–471, Right iris = 473–476
    h, w, _ = frame.shape
    left_iris = np.array([(face_landmarks.landmark[i].x * w,
                           face_landmarks.landmark[i].y * h) for i in range(468, 472)])
    right_iris = np.array([(face_landmarks.landmark[i].x * w,
                            face_landmarks.landmark[i].y * h) for i in range(473, 477)])

    # Eye corners (landmarks: left eye = 33 & 133, right eye = 362 & 263)
    left_eye_outer = np.array([face_landmarks.landmark[33].x * w,
                               face_landmarks.landmark[33].y * h])
    left_eye_inner = np.array([face_landmarks.landmark[133].x * w,
                               face_landmarks.landmark[133].y * h])

    right_eye_outer = np.array([face_landmarks.landmark[362].x * w,
                                face_landmarks.landmark[362].y * h])
    right_eye_inner = np.array([face_landmarks.landmark[263].x * w,
                                face_landmarks.landmark[263].y * h])

    # Iris centers
    left_center = np.mean(left_iris, axis=0)
    right_center = np.mean(right_iris, axis=0)

    # Ratios → where iris is relative to eye corners
    left_ratio = (left_center[0] - left_eye_outer[0]) / (left_eye_inner[0] - left_eye_outer[0])
    right_ratio = (right_center[0] - right_eye_outer[0]) / (right_eye_inner[0] - right_eye_outer[0])
    avg_ratio = (left_ratio + right_ratio) / 2

    # Decision
    if avg_ratio < 0.35:   # looking LEFT
        return "left"
    elif avg_ratio > 0.65: # looking RIGHT
        return "right"
    else:
        return "center"


def detect_abnormal_gaze(shape, frame):
    """
    Detect abnormal gaze ONLY when eyes move left/right.
    Ignores up/down.
    """
    global prev_gaze

    gaze_direction = get_gaze_direction(shape, frame)

    if gaze_direction in ["left", "right"] and prev_gaze == gaze_direction:
        return True

    prev_gaze = gaze_direction
    return False


def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


# def get_gaze_direction(shape):
#     left_eye_top = shape[37][1]
#     left_eye_bottom = shape[41][1]
#     right_eye_top = shape[44][1]
#     right_eye_bottom = shape[46][1]
#     eye_openness = (left_eye_bottom - left_eye_top + right_eye_bottom - right_eye_top) / 2
#
#     if eye_openness < 5:
#         return "closed"
#
#     left_iris_y = (shape[37][1] + shape[40][1]) / 2
#     right_iris_y = (shape[43][1] + shape[46][1]) / 2
#     avg_iris_y = (left_iris_y + right_iris_y) / 2
#     eye_center_y = (shape[37][1] + shape[41][1] + shape[44][1] + shape[46][1]) / 4
#
#     if avg_iris_y > eye_center_y + 5:
#         return "down"
#     elif avg_iris_y < eye_center_y - 5:
#         return "up"
#     else:
#         return "center"

# ==============================
# Hand & Passing Detection (with visualization)
# ==============================
def detect_item_passing(frame, faces):
    global yolo_model
    load_yolo_model()

    results = yolo_model(frame, verbose=False)
    passing_detected = False
    message = None

    # Extract face bounding boxes
    face_boxes = []
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_boxes.append((x, y, x + w, y + h))

    hands = []
    small_objects = []

    # Detect hands & small objects
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # YOLO COCO doesn’t have "hand", so approximate:
                # class_id 0 = person → parts often include hands
                # 67 = phone, 73 = book, 75 = remote (proxy for "small items")
                if class_id == 0:
                    hands.append((x1, y1, x2, y2, conf))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, "Hand/Person", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                elif class_id in [67, 73, 75]:
                    small_objects.append((x1, y1, x2, y2, conf))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, "Object", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Match objects with hands (object must be near a hand)
    for obj in small_objects:
        ox1, oy1, ox2, oy2, conf = obj
        object_center = ((ox1 + ox2) // 2, (oy1 + oy2) // 2)

        near_hand = any(
            (hx1 < object_center[0] < hx2 and hy1 < object_center[1] < hy2)
            for hx1, hy1, hx2, hy2, _ in hands
        )

        if not near_hand:
            continue  # Ignore objects not held in hand

        # Check which faces this object overlaps
        owners = []
        for i, (fx1, fy1, fx2, fy2) in enumerate(face_boxes):
            if fx1 - 50 < object_center[0] < fx2 + 50 and fy1 - 50 < object_center[1] < fy2 + 50:
                owners.append(i)

        # If shared between two students → suspicious passing
        if len(owners) >= 2:
            passing_detected = True
            message = "Suspicious hand movement - Item passing detected"
            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 0, 255), 3)
            cv2.putText(frame, "Item Passing", (ox1, oy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return passing_detected, message




# ==============================
# Main detection function
# ==============================
def detect_cheating_behaviors(frame):
    global missed_frames
    load_models()
    load_yolo_model()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    behaviors = {
        "head_turning": False,
        "abnormal_eye_movement": False,
        "multiple_faces": False,
        "person_proximity": False,
        "prohibited_objects": False,
        "lip_movement": False,
        "item_passing": False,
        "message": "Normal behavior",
        "detected_objects": [],
        "log_entries": []  # NEW: Store log entries for this frame
    }

    # ==============================
    # Multiple faces
    # ==============================
    if detect_multiple_faces(faces, frame):
        behaviors["multiple_faces"] = True
        behaviors["message"] = "Multiple faces detected - potential impersonation"
        # Log multiple faces detection
        log_entry = log_detection("Multiple", "Impersonation", behaviors["message"])
        behaviors["log_entries"].append(log_entry)

    # ==============================
    # Proximity
    # ==============================
    if detect_person_proximity(faces, gray):
        behaviors["person_proximity"] = True
        behaviors["message"] = "Suspicious closeness between examinees"
        # Log proximity detection
        log_entry = log_detection("Multiple", "Proximity", behaviors["message"])
        behaviors["log_entries"].append(log_entry)

    # ==============================
    # Prohibited objects
    # ==============================
    detected_objects = detect_prohibited_objects(frame)
    if detected_objects:
        behaviors["prohibited_objects"] = True
        behaviors["detected_objects"] = detected_objects
        behaviors["message"] = "Prohibited object detected"
        # Log prohibited objects
        for obj in detected_objects:
            log_entry = log_detection("Unknown", "Prohibited Object",
                                      f"Cell phone detected (conf: {obj['confidence']:.2f})")
            behaviors["log_entries"].append(log_entry)

    # ==============================
    # Item Passing Detection
    # ==============================
    passing, passing_message = detect_item_passing(frame, faces)
    if passing:
        behaviors["message"] = passing_message
        behaviors["item_passing"] = True
        # Log item passing
        log_entry = log_detection("Multiple", "Item Passing", passing_message)
        behaviors["log_entries"].append(log_entry)

    # ==============================
    # Process each face (with student IDs)
    # ==============================
    for i, face in enumerate(faces):
        student_id = f"Student {i + 1}"
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        head_turned, turn_direction = detect_head_turn(shape, frame)
        abnormal_gaze = detect_abnormal_gaze(shape, frame)
        talking = detect_lip_movement(shape, frame)

        if head_turned:
            behaviors["head_turning"] = True
            behaviors["message"] = f"Excessive head turning ({turn_direction})"
            # Log head turning with student ID
            log_entry = log_detection(student_id, "Head Turning", behaviors["message"])
            behaviors["log_entries"].append(log_entry)

        if abnormal_gaze:
            behaviors["abnormal_eye_movement"] = True
            behaviors["message"] = "Abnormal eye movement detected"
            # Log abnormal gaze with student ID
            log_entry = log_detection(student_id, "Abnormal Gaze", behaviors["message"])
            behaviors["log_entries"].append(log_entry)

        if talking:
            behaviors["lip_movement"] = True
            behaviors["message"] = "Lip movement detected - potential talking"
            # Log talking with student ID
            log_entry = log_detection(student_id, "Talking", behaviors["message"])
            behaviors["log_entries"].append(log_entry)

        # Draw landmarks and bounding boxes
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        for (x, y) in shape[48:68]:
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add student ID text above bounding box
        cv2.putText(frame, student_id, (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if (behaviors["head_turning"] or behaviors["abnormal_eye_movement"] or
                behaviors["multiple_faces"] or behaviors["person_proximity"] or
                behaviors["prohibited_objects"] or behaviors["lip_movement"]):
            cv2.putText(frame, behaviors["message"], (face.left(), face.top() - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # ==============================
    # Draw prohibited objects
    # ==============================
    for obj in behaviors["detected_objects"]:
        x1, y1, x2, y2 = obj["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Phone: {obj['confidence']:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame, behaviors

