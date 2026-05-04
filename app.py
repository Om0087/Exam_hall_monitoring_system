import eventlet

eventlet.monkey_patch()

import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from utils.detection_utils import detect_cheating_behaviors
from utils.file_utils import allowed_file
import cv2
import base64
import time
import threading
from queue import Queue
import numpy as np
from flask import send_file
from datetime import datetime
from utils.logging_utils import clear_logs
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")  # Changed to eventlet

from utils.detection_utils import load_models, load_yolo_model

# Make sure dlib & YOLO models are loaded at startup
load_models()
load_yolo_model()

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Video processing queues and flags
video_queues = {}
processing_flags = {}
live_captures = {}  # NEW: Track live camera captures


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/download_logs')
def download_logs():
    """Download CSV logs"""
    try:
        return send_file('detection_logs.csv',
                         as_attachment=True,
                         download_name=f'detection_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    except FileNotFoundError:
        return "No logs available yet", 404


@app.route('/live_camera')
def live_camera():
    return render_template('live_camera.html')


@app.route('/about')
def about_us():
    return render_template('about.html')


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
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            return jsonify({
                'status': 'success',
                'filename': filename,
                'redirect': url_for('upload_video', filename=filename)
            })

    filename = request.args.get('filename', None)
    return render_template('upload_video.html', filename=filename)


def process_video_frames(filepath, sid):
    """Process video frames in a separate thread"""
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            socketio.emit('video_error', {'message': 'Could not open video file'}, room=sid, namespace='/upload')
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate target processing rate (adjust based on performance)
        target_fps = min(fps, 10)  # Limit to 10 FPS for processing
        frame_skip = max(1, int(fps / target_fps))

        print(f"Video info: {total_frames} frames, {fps} FPS, skipping {frame_skip - 1} frames")

        frame_count = 0
        processed_count = 0

        while cap.isOpened() and processing_flags.get(sid, True):
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames to maintain reasonable speed
            if frame_count % frame_skip != 0:
                continue

            processed_count += 1
            progress = min(100, int((frame_count / total_frames) * 100))

            try:
                # Detect cheating behaviors (with timeout)
                processed_frame, behaviors = detect_cheating_behaviors(frame)

                # Resize frame for faster transmission
                processed_frame = cv2.resize(processed_frame, (800, 600))

                # Convert frame to JPEG with lower quality for faster transmission
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                # Send frame and detection data
                socketio.emit('uploaded_feed', {
                    'frame': frame_base64,
                    'behaviors': behaviors,
                    'progress': progress,
                    'frame_number': frame_count,
                    'total_frames': total_frames
                }, room=sid, namespace='/upload')

                # Small delay to prevent overwhelming the client
                eventlet.sleep(0.01)

            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                continue

        cap.release()
        socketio.emit('video_ended', room=sid, namespace='/upload')
        print(f"Video processing completed for session {sid}")

    except Exception as e:
        print(f"Error in video processing: {e}")
        socketio.emit('video_error', {'message': f'Error processing video: {str(e)}'}, room=sid, namespace='/upload')


def process_live_feed(camera_index, sid):
    """Process live camera feed in separate thread"""
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            socketio.emit('camera_error', {'message': 'Could not access camera'}, room=sid, namespace='/live')
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)

        live_captures[sid] = cap  # Store capture object

        while cap.isOpened() and processing_flags.get(sid, False):
            ret, frame = cap.read()
            if not ret:
                socketio.emit('camera_error', {'message': 'Failed to read frame from camera'}, room=sid,
                              namespace='/live')
                break

            try:
                # Detect cheating behaviors
                processed_frame, behaviors = detect_cheating_behaviors(frame)

                # Resize for faster transmission
                processed_frame = cv2.resize(processed_frame, (640, 480))

                # Convert frame to JPEG
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                # Send frame and detection data
                socketio.emit('live_feed', {
                    'frame': frame_base64,
                    'behaviors': behaviors
                }, room=sid, namespace='/live')

            except Exception as e:
                print(f"Error in live feed: {e}")
                continue

            eventlet.sleep(0.067)  # ~15 FPS

        # Cleanup
        if sid in live_captures:
            live_captures[sid].release()
            del live_captures[sid]

    except Exception as e:
        print(f"Error in live feed processing: {e}")
        socketio.emit('camera_error', {'message': f'Camera error: {str(e)}'}, room=sid, namespace='/live')


# SocketIO Event Handlers for Upload Namespace
@socketio.on('connect', namespace='/upload')
def handle_upload_connect():
    print('Client connected to upload namespace')
    video_queues[request.sid] = Queue()
    processing_flags[request.sid] = True


@socketio.on('disconnect', namespace='/upload')
def handle_upload_disconnect():
    print('Client disconnected from upload namespace')
    processing_flags[request.sid] = False
    if request.sid in video_queues:
        del video_queues[request.sid]
    if request.sid in processing_flags:
        del processing_flags[request.sid]


@socketio.on('start_uploaded_feed', namespace='/upload')
def handle_start_uploaded_feed(data):
    # Clear previous logs when starting new session
    clear_logs()

    filename = data.get('filename')
    sid = request.sid

    if not filename:
        emit('video_error', {'message': 'No filename provided'})
        return

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(filepath):
        emit('video_error', {'message': 'File not found'})
        return

    # Start video processing in a separate thread
    processing_flags[sid] = True
    thread = threading.Thread(target=process_video_frames, args=(filepath, sid))
    thread.daemon = True
    thread.start()


@socketio.on('pause_uploaded_feed', namespace='/upload')
def handle_pause_uploaded_feed():
    processing_flags[request.sid] = False


@socketio.on('resume_uploaded_feed', namespace='/upload')
def handle_resume_uploaded_feed():
    processing_flags[request.sid] = True


# SocketIO Event Handlers for Live Namespace
@socketio.on('connect', namespace='/live')
def handle_live_connect():
    print('Client connected to live feed namespace')
    processing_flags[request.sid] = False


@socketio.on('disconnect', namespace='/live')
def handle_live_disconnect():
    print('Client disconnected from live feed namespace')
    processing_flags[request.sid] = False
    if request.sid in live_captures:
        live_captures[request.sid].release()
        del live_captures[request.sid]
    if request.sid in processing_flags:
        del processing_flags[request.sid]


@socketio.on('start_live_feed', namespace='/live')
def handle_start_live_feed(data):
    # Clear previous logs when starting new session
    clear_logs()

    camera_index = data.get('camera_index', 0)
    sid = request.sid

    # Stop any existing feed
    processing_flags[sid] = False
    eventlet.sleep(0.1)  # Brief pause to ensure cleanup

    # Start new feed
    processing_flags[sid] = True
    thread = threading.Thread(target=process_live_feed, args=(camera_index, sid))
    thread.daemon = True
    thread.start()


@socketio.on('stop_live_feed', namespace='/live')
def handle_stop_live_feed():
    sid = request.sid
    processing_flags[sid] = False
    print('Live feed stopped by client')


if __name__ == '__main__':
    print("✅ App setup complete. Starting server...")
    socketio.run(app, debug=True, use_reloader=False, port=0, allow_unsafe_werkzeug=True)