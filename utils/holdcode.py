import cv2
import dlib
import numpy as np
from imutils import face_utils

# Constants
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
HEAD_TURN_THRESHOLD = 20
# In detection_utils.py, add these constants near the top
MULTIPLE_FACE_THRESHOLD = 2  # Number of faces to trigger impersonation alert
PROXIMITY_THRESHOLD = 100    # Pixel distance to trigger proximity alert

# Lazy load models
detector = None
predictor = None

# State
eye_counter = 0
prev_gaze = "center"


def load_models():
    global detector, predictor
    if detector is None or predictor is None:
        print("[INFO] Loading dlib models...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        print("[INFO] Models loaded successfully.")

# In detection_utils.py, update the detect_cheating_behaviors function
# Add these imports at the top of detection_utils.py
from ultralytics import YOLO
import numpy as np

# Add these constants after the existing ones
LIP_MOVEMENT_THRESHOLD = 0.05  # Threshold for lip movement detection
LIP_MOVEMENT_CONSEC_FRAMES = 5  # Number of consecutive frames for talking detection

# Lazy load YOLO model
yolo_model = None

# State for lip movement detection
lip_movement_history = []
prev_lip_distance = 0
talk_counter = 0


def load_yolo_model():
    global yolo_model
    if yolo_model is None:
        print("[INFO] Loading YOLO model for object detection...")
        # Using a general YOLOv8 model that can detect phones, etc.
        yolo_model = YOLO('yolov8n.pt')  # Using nano version for speed
        print("[INFO] YOLO model loaded successfully.")


def detect_prohibited_objects(frame):
    """
    Detect prohibited objects like mobile phones, smart watches, etc.
    Returns list of detected objects and their positions
    """
    global yolo_model
    load_yolo_model()

    # Define prohibited object classes (COCO dataset classes)
    # 67: cell phone, 73: laptop, 64: mouse, 66: keyboard, 72: TV, 74: remote
    # We'll focus on cell phones primarily but include other electronics
    prohibited_classes = [67]  # Cell phone

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


def detect_lip_movement(shape, frame):
    """
    Detect lip movement to identify potential talking
    Returns True if talking is detected
    """
    global lip_movement_history, prev_lip_distance, talk_counter

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
    lip_movement_history.append(mar)
    if len(lip_movement_history) > 10:  # Keep last 10 frames
        lip_movement_history.pop(0)

    # Calculate movement (change in MAR)
    if len(lip_movement_history) > 1:
        movement = abs(lip_movement_history[-1] - lip_movement_history[-2])

        # Check if movement exceeds threshold
        if movement > LIP_MOVEMENT_THRESHOLD:
            talk_counter += 1
            if talk_counter >= LIP_MOVEMENT_CONSEC_FRAMES:
                talk_counter = 0
                return True
        else:
            talk_counter = max(0, talk_counter - 1)

    return False


# Update the detect_cheating_behaviors function to include the new detections
def detect_cheating_behaviors(frame):
    global eye_counter, prev_gaze
    load_models()  # Ensure models are loaded

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    behaviors = {
        'head_turning': False,
        'abnormal_eye_movement': False,
        'multiple_faces': False,
        'person_proximity': False,
        'prohibited_objects': False,
        'lip_movement': False,
        'message': "Normal behavior",
        'detected_objects': []
    }

    # Check for multiple faces
    if detect_multiple_faces(faces, frame):
        behaviors['multiple_faces'] = True
        behaviors['message'] = "Multiple faces detected - potential impersonation"

    # Check for person proximity
    if detect_person_proximity(faces, gray):
        behaviors['person_proximity'] = True
        behaviors['message'] = "Suspicious closeness between examinees"

    # Check for prohibited objects
    detected_objects = detect_prohibited_objects(frame)
    if detected_objects:
        behaviors['prohibited_objects'] = True
        behaviors['detected_objects'] = detected_objects
        behaviors['message'] = "Prohibited object detected"

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Head pose estimation
        head_turned, turn_direction = detect_head_turn(shape, frame)

        # Eye gaze detection
        abnormal_gaze = detect_abnormal_gaze(shape, frame)

        # Lip movement detection (talking)
        talking = detect_lip_movement(shape, frame)

        if head_turned:
            behaviors['head_turning'] = True
            behaviors['message'] = f"Excessive head turning ({turn_direction})"

        if abnormal_gaze:
            behaviors['abnormal_eye_movement'] = True
            behaviors['message'] = "Abnormal eye movement detected"

        if talking:
            behaviors['lip_movement'] = True
            behaviors['message'] = "Lip movement detected - potential talking"

        # Draw facial landmarks and annotations
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Draw lip landmarks in a different color
        for (x, y) in shape[48:68]:  # Lip points
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

        # Draw face bounding boxes
        x, y, w, h = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if (behaviors['head_turning'] or behaviors['abnormal_eye_movement'] or
                behaviors['multiple_faces'] or behaviors['person_proximity'] or
                behaviors['prohibited_objects'] or behaviors['lip_movement']):
            cv2.putText(frame, behaviors['message'], (face.left(), face.top() - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw lines between faces if proximity is detected
    if behaviors['person_proximity']:
        # Get centroids of all faces
        centroids = []
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            centroid_x = int(x + w / 2)
            centroid_y = int(y + h / 2)
            centroids.append((centroid_x, centroid_y))
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

        # Draw lines between all pairs of faces
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                cv2.line(frame, centroids[i], centroids[j], (0, 0, 255), 2)

    # Draw bounding boxes for detected objects
    for obj in behaviors['detected_objects']:
        x1, y1, x2, y2 = obj['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Phone: {obj['confidence']:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame, behaviors

# def detect_cheating_behaviors(frame):
#     global eye_counter, prev_gaze
#     load_models()  # Ensure models are loaded
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray, 0)
#
#     behaviors = {
#         'head_turning': False,
#         'abnormal_eye_movement': False,
#         'message': "Normal behavior"
#     }
#
#     for face in faces:
#         shape = predictor(gray, face)
#         shape = face_utils.shape_to_np(shape)
#
#         # Head pose estimation
#         head_turned, turn_direction = detect_head_turn(shape, frame)
#
#         # Eye gaze detection
#         abnormal_gaze = detect_abnormal_gaze(shape, frame)
#
#         if head_turned:
#             behaviors['head_turning'] = True
#             behaviors['message'] = f"Excessive head turning ({turn_direction})"
#
#         if abnormal_gaze:
#             behaviors['abnormal_eye_movement'] = True
#             behaviors['message'] = "Abnormal eye movement detected"
#
#         # Draw facial landmarks and annotations
#         for (x, y) in shape:
#             cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
#
#         if behaviors['head_turning'] or behaviors['abnormal_eye_movement']:
#             cv2.putText(frame, behaviors['message'], (face.left(), face.top() - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#
#     return frame, behaviors


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


def detect_person_proximity(faces, frame):
    """
    Detect if people are too close to each other
    Returns True if any two faces are closer than PROXIMITY_THRESHOLD
    """
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            # Get center points of both faces
            face_i_center = (faces[i].left() + faces[i].width() // 2,
                             faces[i].top() + faces[i].height() // 2)
            face_j_center = (faces[j].left() + faces[j].width() // 2,
                             faces[j].top() + faces[j].height() // 2)

            # Calculate Euclidean distance between centers
            distance = np.sqrt((face_i_center[0] - face_j_center[0]) ** 2 +
                               (face_i_center[1] - face_j_center[1]) ** 2)

            if distance < PROXIMITY_THRESHOLD:
                return True
    return False


def draw_proximity_lines(faces, frame):
    """
    Draw lines between faces that are close to each other
    """
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            # Get center points of both faces
            face_i_center = (faces[i].left() + faces[i].width() // 2,
                             faces[i].top() + faces[i].height() // 2)
            face_j_center = (faces[j].left() + faces[j].width() // 2,
                             faces[j].top() + faces[j].height() // 2)

            # Calculate Euclidean distance between centers
            distance = np.sqrt((face_i_center[0] - face_j_center[0]) ** 2 +
                               (face_i_center[1] - face_j_center[1]) ** 2)

            # Draw line if faces are close
            if distance < PROXIMITY_THRESHOLD * 1.5:  # Slightly larger threshold for visualization
                color = (0, 0, 255) if distance < PROXIMITY_THRESHOLD else (0, 255, 255)
                thickness = 2 if distance < PROXIMITY_THRESHOLD else 1
                cv2.line(frame, face_i_center, face_j_center, color, thickness)

                # Draw distance text
                mid_point = ((face_i_center[0] + face_j_center[0]) // 2,
                             (face_i_center[1] + face_j_center[1]) // 2)
                cv2.putText(frame, f"{int(distance)}", mid_point,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

                def detect_cheating_behaviors(frame):
                    global eye_counter, prev_gaze, lip_movement_history, prev_lip_distance, talk_counter
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
                        'message': "Normal behavior",
                        'detected_objects': []
                    }

                    # Check for multiple faces (impersonation detection)
                    if detect_multiple_faces(faces, frame):
                        behaviors['multiple_faces'] = True
                        behaviors['message'] = "Multiple faces detected - potential impersonation"

                    # Check for person proximity (suspicious closeness)
                    if detect_person_proximity(faces, gray):
                        behaviors['person_proximity'] = True
                        behaviors['message'] = "Suspicious closeness between examinees"

                    # Check for prohibited objects (phones, smart watches, etc.)
                    detected_objects = detect_prohibited_objects(frame)
                    if detected_objects:
                        behaviors['prohibited_objects'] = True
                        behaviors['detected_objects'] = detected_objects
                        behaviors['message'] = "Prohibited object detected"

                    # Process each detected face for individual behaviors
                    for face in faces:
                        shape = predictor(gray, face)
                        shape = face_utils.shape_to_np(shape)

                        # Head pose estimation (head turning detection)
                        head_turned, turn_direction = detect_head_turn(shape, frame)

                        # Eye gaze detection (abnormal eye movements)
                        abnormal_gaze = detect_abnormal_gaze(shape, frame)

                        # Lip movement detection (talking detection)
                        talking = detect_lip_movement(shape, frame)

                        # Update behaviors based on detections
                        if head_turned:
                            behaviors['head_turning'] = True
                            behaviors['message'] = f"Excessive head turning ({turn_direction})"

                        if abnormal_gaze:
                            behaviors['abnormal_eye_movement'] = True
                            behaviors['message'] = "Abnormal eye movement detected"

                        if talking:
                            behaviors['lip_movement'] = True
                            behaviors['message'] = "Lip movement detected - potential talking"

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

                    # Draw lines between faces if proximity is detected (red lines)
                    if behaviors['person_proximity']:
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

                # Helper functions required for detect_cheating_behaviors

    def detect_multiple_faces(faces, frame):
                    """
                    Detect if multiple faces are present in the frame
                    Returns True if potential impersonation is detected
                    """
                    if len(faces) > MULTIPLE_FACE_THRESHOLD:
                        return True
                    return False

    def detect_person_proximity(faces, gray):
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

                    # Define prohibited object classes (COCO dataset classes)
                    # 67: cell phone, 73: laptop, 64: mouse, 66: keyboard, 72: TV, 74: remote
                    # We'll focus on cell phones primarily but include other electronics
                    prohibited_classes = [67]  # Cell phone

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

def detect_lip_movement(shape, frame):
                        """
                        Detect lip movement to identify potential talking
                        Returns True if talking is detected
                        """
                        global lip_movement_history, prev_lip_distance, talk_counter

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
                        lip_movement_history.append(mar)
                        if len(lip_movement_history) > 10:  # Keep last 10 frames
                            lip_movement_history.pop(0)

                        # Calculate movement (change in MAR)
                        if len(lip_movement_history) > 1:
                            movement = abs(lip_movement_history[-1] - lip_movement_history[-2])

                            # Check if movement exceeds threshold
                            if movement > LIP_MOVEMENT_THRESHOLD:
                                talk_counter += 1
                                if talk_counter >= LIP_MOVEMENT_CONSEC_FRAMES:
                                    talk_counter = 0
                                    return True
                            else:
                                talk_counter = max(0, talk_counter - 1)

                        return False

