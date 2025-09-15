import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
from ultralytics import YOLO
import math

# Constants
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
HEAD_TURN_THRESHOLD = 20
MULTIPLE_FACE_THRESHOLD = 2  # Number of faces to trigger impersonation alert
PROXIMITY_THRESHOLD = 100  # Pixel distance to trigger proximity alert
LIP_MOVEMENT_THRESHOLD = 0.05  # Threshold for lip movement detection
LIP_MOVEMENT_CONSEC_FRAMES = 5  # Number of consecutive frames for talking detection
TALKING_PROXIMITY_THRESHOLD = 150  # Distance threshold for detecting people talking to each other

# Lazy load models
detector = None
predictor = None
yolo_model = None

# State variables
eye_counter = 0
prev_gaze = "center"
lip_movement_history = {}
prev_lip_distance = {}
talk_counter = {}
conversation_history = []


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
        yolo_model = YOLO('yolov8n.pt')  # Using nano version for speed
        print("[INFO] YOLO model loaded successfully.")


def detect_multiple_faces(faces, frame):
    """
    Detect if multiple faces are present in the frame
    Returns True if potential impersonation is detected
    """
    if len(faces) >= MULTIPLE_FACE_THRESHOLD:
        return True
    return False


def detect_person_proximity(faces, frame):
    """
    Detect if people are too close to each other
    Returns True if suspicious closeness is detected
    """
    if len(faces) < 2:
        return False

    # Get centroids of all faces
    centroids = []
    for face in faces:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        centroid_x = x + w / 2
        centroid_y = y + h / 2
        centroids.append((centroid_x, centroid_y))

    # Check distances between all pairs of faces
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            # Calculate Euclidean distance between centroids
            d = dist.euclidean(centroids[i], centroids[j])
            if d < PROXIMITY_THRESHOLD:
                return True
    return False


def detect_prohibited_objects(frame):
    """
    Detect prohibited objects like mobile phones, smart watches, etc.
    Returns list of detected objects and their positions
    """
    global yolo_model
    load_yolo_model()

    # Define prohibited object classes (COCO dataset classes)
    prohibited_classes = {
        67: "cell phone",  # actual phone
        63: "laptop",  # can represent tablets too
        73: "book",  # could act as chit/paper proxy
        27: "backpack",  # cheating material storage
        26: "handbag" }  # Cell phone

    detected_objects = []

    # Run YOLO inference
    results = yolo_model(frame, verbose=False)

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id in prohibited_classes:
                    confidence = float(box.conf[0])
                    if confidence > 0.5:  # Confidence threshold
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detected_objects.append({
                            'class': class_id,
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })

    return detected_objects


def detect_lip_movement(shape, face_id):
    """
    Detect lip movement to identify potential talking
    Returns True if talking is detected
    """
    global lip_movement_history, prev_lip_distance, talk_counter

    # Initialize history for new faces
    if face_id not in lip_movement_history:
        lip_movement_history[face_id] = []
        talk_counter[face_id] = 0
        prev_lip_distance[face_id] = 0

    # Get lip landmarks (points 48-68 in the 68-point model)
    lips = shape[48:68]

    # Calculate mouth aspect ratio (MAR)
    # Horizontal distance
    A = np.linalg.norm(lips[6] - lips[0])  # Corner to corner
    # Vertical distances
    B1 = np.linalg.norm(lips[2] - lips[10])  # Top to bottom
    B2 = np.linalg.norm(lips[4] - lips[8])  # Top to bottom

    # Average vertical distance
    B = (B1 + B2) / 2.0

    # Avoid division by zero
    if A == 0:
        return False

    # Mouth aspect ratio
    mar = B / A

    # Store in history
    lip_movement_history[face_id].append(mar)
    if len(lip_movement_history[face_id]) > 10:  # Keep last 10 frames
        lip_movement_history[face_id].pop(0)

    # Calculate movement (change in MAR)
    if len(lip_movement_history[face_id]) > 1:
        movement = abs(lip_movement_history[face_id][-1] - lip_movement_history[face_id][-2])

        # Check if movement exceeds threshold
        if movement > LIP_MOVEMENT_THRESHOLD:
            talk_counter[face_id] += 1
            if talk_counter[face_id] >= LIP_MOVEMENT_CONSEC_FRAMES:
                talk_counter[face_id] = 0
                return True
        else:
            talk_counter[face_id] = max(0, talk_counter[face_id] - 1)

    return False


def detect_head_turn(shape, frame):
    # Get the 2D facial landmarks
    image_points = np.array([
        shape[30],  # Nose tip
        shape[8],  # Chin
        shape[36],  # Left eye left corner
        shape[45],  # Right eye right corner
        shape[48],  # Left mouth corner
        shape[54]  # Right mouth corner
    ], dtype="double")

    # 3D model points
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # Camera internals
    size = frame.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    # Solve for pose
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Calculate rotation angles
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    # Get yaw angle (head turning left/right)
    yaw = angles[1]

    # Determine if head is turned
    if yaw < -HEAD_TURN_THRESHOLD:
        return True, "left"
    elif yaw > HEAD_TURN_THRESHOLD:
        return True, "right"
    else:
        return False, "center"


def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # Return the eye aspect ratio
    return ear


def get_gaze_direction(shape):
    # Simple gaze detection based on relative position of eye landmarks
    left_eye_top = shape[37][1]
    left_eye_bottom = shape[41][1]
    right_eye_top = shape[44][1]
    right_eye_bottom = shape[46][1]

    eye_openness = (left_eye_bottom - left_eye_top + right_eye_bottom - right_eye_top) / 2

    if eye_openness < 5:  # Threshold for closed eyes
        return "closed"

    # Compare iris position (simplified)
    left_iris_y = (shape[37][1] + shape[40][1]) / 2
    right_iris_y = (shape[43][1] + shape[46][1]) / 2
    avg_iris_y = (left_iris_y + right_iris_y) / 2

    eye_center_y = (shape[37][1] + shape[41][1] + shape[44][1] + shape[46][1]) / 4

    if avg_iris_y > eye_center_y + 5:
        return "down"
    elif avg_iris_y < eye_center_y - 5:
        return "up"
    else:
        return "center"


def detect_abnormal_gaze(shape, frame):
    global eye_counter, prev_gaze

    # Extract left and right eye coordinates
    left_eye = shape[42:48]
    right_eye = shape[36:42]

    # Calculate eye aspect ratios
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    # Average the eye aspect ratio
    ear = (left_ear + right_ear) / 2.0

    # Check for abnormal gaze (looking down or prolonged blink)
    if ear < EYE_AR_THRESH:
        eye_counter += 1
        if eye_counter >= EYE_AR_CONSEC_FRAMES:
            eye_counter = 0
            return True
    else:
        eye_counter = 0

    # Additional gaze detection based on eye landmarks
    gaze_direction = get_gaze_direction(shape)

    if gaze_direction == "down" and prev_gaze == "down":
        return True

    prev_gaze = gaze_direction
    return False


def detect_conversation(faces, talking_faces, frame):
    """
    Detect if two or more people are talking to each other
    Returns True if conversation is detected
    """
    if len(talking_faces) < 2:
        return False

    # Get centroids of talking faces
    talking_centroids = []
    for i in talking_faces:
        face = faces[i]
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        centroid_x = x + w / 2
        centroid_y = y + h / 2
        talking_centroids.append((centroid_x, centroid_y, i))

    # Check if any two talking people are close enough to be having a conversation
    for i in range(len(talking_centroids)):
        for j in range(i + 1, len(talking_centroids)):
            # Calculate Euclidean distance between centroids
            d = dist.euclidean(talking_centroids[i][:2], talking_centroids[j][:2])
            if d < TALKING_PROXIMITY_THRESHOLD:
                # Check if they're facing each other (simplified)
                # For a real implementation, you'd need head pose estimation for each person
                return True

    return False


def detect_cheating_behaviors(frame):
    global eye_counter, prev_gaze, lip_movement_history, prev_lip_distance, talk_counter, conversation_history
    load_models()  # Ensure models are loaded
    load_yolo_model()  # Ensure YOLO model is loaded

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    behaviors = {
        'head_turning': False,
        'abnormal_eye_movement': False,
        'multiple_faces': False,
        'person_proximity': False,
        'prohibited_objects': False,
        'lip_movement': False,
        'conversation': False,
        'message': "Normal behavior",
        'detected_objects': [],
        'talking_faces': []
    }

    # Check for multiple faces (impersonation detection)
    if detect_multiple_faces(faces, frame):
        behaviors['multiple_faces'] = True
        behaviors['message'] = "Multiple faces detected - potential impersonation"

    # Check for person proximity (suspicious closeness)
    if detect_person_proximity(faces, frame):
        behaviors['person_proximity'] = True
        behaviors['message'] = "Suspicious closeness between examinees"

    # Check for prohibited objects (phones, smart watches, etc.)
    detected_objects = detect_prohibited_objects(frame)
    if detected_objects:
        behaviors['prohibited_objects'] = True
        behaviors['detected_objects'] = detected_objects
        behaviors['message'] = "Prohibited object detected"

    # Process each detected face for individual behaviors
    talking_faces = []
    for i, face in enumerate(faces):
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Head pose estimation (head turning detection)
        head_turned, turn_direction = detect_head_turn(shape, frame)

        # Eye gaze detection (abnormal eye movements)
        abnormal_gaze = detect_abnormal_gaze(shape, frame)

        # Lip movement detection (talking detection)
        talking = detect_lip_movement(shape, i)
        if talking:
            talking_faces.append(i)
            behaviors['lip_movement'] = True
            behaviors['talking_faces'].append(i)

        # Update behaviors based on detections
        if head_turned:
            behaviors['head_turning'] = True
            behaviors['message'] = f"Excessive head turning ({turn_direction})"

        if abnormal_gaze:
            behaviors['abnormal_eye_movement'] = True
            behaviors['message'] = "Abnormal eye movement detected"

        # Draw facial landmarks and annotations
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Draw lip landmarks in a different color (blue)
        for (x, y) in shape[48:68]:  # Lip points (48-67 in 68-point model)
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

        # Draw face bounding boxes (green)
        x, y, w, h = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display warning message if any cheating behavior is detected
        if (behaviors['head_turning'] or behaviors['abnormal_eye_movement'] or
                behaviors['multiple_faces'] or behaviors['person_proximity'] or
                behaviors['prohibited_objects'] or behaviors['lip_movement']):
            cv2.putText(frame, behaviors['message'], (face.left(), face.top() - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Check for conversations (multiple people talking)
    if len(talking_faces) >= 2:
        if detect_conversation(faces, talking_faces, frame):
            behaviors['conversation'] = True
            behaviors['message'] = "Multiple people talking - potential conversation"

            # Draw conversation indicators
            for i in talking_faces:
                face = faces[i]
                x, y, w, h = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
                cv2.putText(frame, "TALKING", (x, y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw lines between faces if proximity is detected (red lines)
    if behaviors['person_proximity'] or behaviors['conversation']:
        centroids = []
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            centroid_x = int(x + w / 2)
            centroid_y = int(y + h / 2)
            centroids.append((centroid_x, centroid_y))
            # Draw centroid points (red)
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

        # Draw lines between all pairs of faces (red)
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                cv2.line(frame, centroids[i], centroids[j], (0, 0, 255), 2)

    # Draw bounding boxes for detected prohibited objects (red boxes)
    for obj in behaviors['detected_objects']:
        x1, y1, x2, y2 = obj['bbox']
        # Draw bounding box (red)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Draw label with confidence score
        label = f"Phone: {obj['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame, behaviors