# logging_utils.py
import csv
import os
from datetime import datetime
import threading

# Thread lock for safe CSV writing
csv_lock = threading.Lock()
def log_detection(student_id, detection_type, message, csv_file="detection_logs.csv"):
    """
    Log detection events to CSV file with timestamp
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create log entry
    log_entry = {
        "timestamp": timestamp,
        "student_id": student_id,
        "detection_type": detection_type,
        "message": message
    }

    # Thread-safe CSV writing
    with csv_lock:
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "student_id", "detection_type", "message"])

            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)

    print(f"📝 Logged: {timestamp} | Student {student_id} | {detection_type} | {message}")
    return log_entry

def clear_logs(csv_file="detection_logs.csv"):
    """Clear existing logs for new session"""
    with csv_lock:
        if os.path.exists(csv_file):
            os.remove(csv_file)