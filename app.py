from flask import Flask, render_template, Response, request, redirect, url_for, session
import cv2
import face_recognition
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Admin credentials (for demo purposes, use a more secure method in production)
ADMIN_CREDENTIALS = {'admin': 'password123'}  # You can change the admin id and password here

# Paths and configurations
data_dir = "known_faces"
app.config['UPLOAD_FOLDER'] = data_dir
app.secret_key = 'supersecretkey'  # Used for session management

# Load known faces
def load_known_faces():
    known_encodings = []
    known_names = []
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names

known_face_encodings, known_face_names = load_known_faces()

# Facial recognition logic
def recognize_face(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            return known_face_names[first_match_index]
    return None

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        admin_id = request.form['admin_id']
        password = request.form['password']
        if ADMIN_CREDENTIALS.get(admin_id) == password:
            session['logged_in'] = True
            return redirect(url_for('admin_panel'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/video_feed')
def video_feed():   
    def gen():
        camera = cv2.VideoCapture(0)
        while True:
            success, frame = camera.read()
            if not success:
                break
            name = recognize_face(frame)
            if name:
                cv2.putText(frame, f"Welcome, {name}!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unrecognized face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        camera.release()
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_face', methods=['GET', 'POST'])
def add_face():
    if 'logged_in' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            global known_face_encodings, known_face_names
            known_face_encodings, known_face_names = load_known_faces()
            return redirect(url_for('admin_panel'))
    return render_template('add_face.html')

@app.route('/admin')
def admin_panel():
    if 'logged_in' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    faces = os.listdir(data_dir) if os.path.exists(data_dir) else []
    return render_template('admin.html', faces=faces)

@app.route('/delete_face/<name>')
def delete_face(name):
    if 'logged_in' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    filepath = os.path.join(data_dir, f"{name}.jpg")
    if os.path.exists(filepath):
        os.remove(filepath)
        global known_face_encodings, known_face_names
        known_face_encodings, known_face_names = load_known_faces()
    return redirect(url_for('admin_panel'))

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('index'))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
