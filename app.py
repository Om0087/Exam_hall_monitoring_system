import eventlet
eventlet.monkey_patch()

import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from utils.detection_utils import detect_cheating_behaviors
from utils.file_utils import allowed_file
import cv2
import base64
import time

from config import Config

app = Flask(__name__)
app.config.from_object(Config)
socketio = SocketIO(app, async_mode='threading')

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/live_camera')
def live_camera():
    return render_template('live_camera.html')


# Add these imports at the top
import os.path
from flask import jsonify


# Update the upload_video route
@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Ensure the uploads directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Return JSON response for AJAX handling
            return jsonify({
                'status': 'success',
                'filename': filename,
                'redirect': url_for('upload_video', filename=filename)
            })

    # For GET requests or when no file is uploaded
    filename = request.args.get('filename', None)
    return render_template('upload_video.html', filename=filename)


# Add these Socket.IO handlers
@socketio.on('connect', namespace='/upload')
def handle_upload_connect():
    print('Client connected to upload namespace')


@socketio.on('disconnect', namespace='/upload')
def handle_upload_disconnect():
    print('Client disconnected from upload namespace')


@socketio.on('start_uploaded_feed', namespace='/upload')
def handle_start_uploaded_feed(data):
    filename = data.get('filename')
    if not filename:
        emit('video_error', {'message': 'No filename provided'})
        return

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(filepath):
        emit('video_error', {'message': 'File not found'})
        return

    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            emit('video_error', {'message': 'Could not open video file'})
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 0.033  # default to ~30fps

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress = int((frame_count / total_frames) * 100)

            # Detect cheating behaviors
            frame, behaviors = detect_cheating_behaviors(frame)

            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Send frame and detection data
            emit('uploaded_feed', {
                'frame': frame_base64,
                'behaviors': behaviors,
                'progress': progress
            })

            eventlet.sleep(frame_delay)  # Maintain original video speed

        cap.release()
        emit('video_ended')
    except Exception as e:
        emit('video_error', {'message': f'Error processing video: {str(e)}'})


@socketio.on('connect', namespace='/live')
def handle_connect():
    print('Client connected to live feed')


@socketio.on('disconnect', namespace='/live')
def handle_disconnect():
    print('Client disconnected from live feed')


@socketio.on('start_live_feed', namespace='/live')
def handle_start_live_feed(data):
    camera_index = data.get('camera_index', 0)
    cap = cv2.VideoCapture(camera_index)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect cheating behaviors
        frame, behaviors = detect_cheating_behaviors(frame)

        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Send frame and detection data
        emit('live_feed', {
            'frame': frame_base64,
            'behaviors': behaviors
        })

        eventlet.sleep(0.05)  # Control frame rate

    cap.release()


@socketio.on('start_uploaded_feed', namespace='/upload')
def handle_start_uploaded_feed(data):
    filename = data['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / fps if fps > 0 else 0.03

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        # Detect cheating behaviors
        frame, behaviors = detect_cheating_behaviors(frame)

        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Send frame and detection data
        emit('uploaded_feed', {
            'frame': frame_base64,
            'behaviors': behaviors
        })

        eventlet.sleep(frame_delay)  # Maintain original video speed

    cap.release()
    emit('video_ended')

#
# if __name__ == '__main__':
#     print("✅ App setup complete. Starting server...")
#     socketio.run(app, debug=False, use_reloader=False)

if __name__ == '__main__':
    print("✅ App setup complete. Starting server...")
    socketio.run(app, debug=False, use_reloader=False, port=5050, allow_unsafe_werkzeug=True)

