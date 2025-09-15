from werkzeug.utils import secure_filename
from flask import current_app
import os

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def get_secure_filename(file):
    filename = secure_filename(file.filename)
    return os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
