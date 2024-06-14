import argparse
import io
import os
import time
from PIL import Image
import datetime
import torch
import cv2
import numpy as np
from flask import Flask, render_template, request, Response, send_from_directory, redirect, session, url_for, flash
from flask_bootstrap import Bootstrap
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
# from flask_wtf import wtforms
from wtforms import Form, StringField, TextAreaField, PasswordField, BooleanField, SubmitField
from wtforms.validators import input_required, length, ValidationError, Email
from email_validator import validate_email
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

app = Flask(__name__)
Bootstrap(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin,db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(20), unique=True)
    password = db.Column(db.String(80))

with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



class LoginForm(FlaskForm):
    username = StringField('username', validators=[input_required(), length(min=4, max=15)],)
    password = PasswordField('password', validators=[input_required(), length(min=8, max=80)],)
    remember = BooleanField("Remember Me")


class RegisterForm(FlaskForm):
    email = StringField('email', validators=[input_required(), Email(message='Invalid Email'), length(max=50)])
    username = StringField('username', validators=[input_required(), length(min=4, max=15)])
    password = PasswordField('password', validators=[input_required(), length(min=8, max=80)])

    def validate_email(self, field):
        if User.query.filter_by(email=field.data).first():
            raise ValidationError('Email address already registered')


@app.route('/login', methods=['Get', 'Post'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('aa'))

        return '<h1>Invalid Username or password</h1>'
        # return '<h1>' + form.username.data + ' ' + form.password.data + '<h1>'

    return render_template('login.html', form=form)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        # No need to check for existing email here, it's already done in the form validation
        hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('signup.html', form=form)


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('aa'))



@app.route("/")
@login_required
def home():
    return render_template('home.html')

@app.route("/home")
@login_required
def aa():
    return render_template('home.html')


@app.route("/index", methods=["GET", "POST"])
@login_required
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            print("Uploaded folder is ", filepath)
            f.save(filepath)

            file_extension = f.filename.rsplit('.', 1)[1].lower()

            if file_extension in ['jpg', 'png']:
                # Perform the detection directly on the saved file
                yolo = YOLO('best.pt')  # Assuming this initializes the model correctly

                # Load the image
                img = Image.open(filepath)

                # Perform detection
                detections = yolo(img, save=True)  # Save the detection results

                # Send back the image with detections
                return display(f.filename)

            elif file_extension == 'mp4':
                video_path = filepath
                cap = cv2.VideoCapture(video_path)

                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_video_path = 'output.mp4'
                out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))

                model = YOLO('best.pt')

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame, save=True)
                    res_plotted = results[0].plot()
                    out.write(res_plotted)

                    if cv2.waitKey(1) == ord('q'):
                        break

                return video_feed()

    return render_template('index.html')




@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = os.path.join(folder_path, latest_subfolder)
    files = os.listdir(directory)
    latest_file = files[0]
    filename = os.path.join(folder_path, latest_subfolder, latest_file)
    file_extension = filename.rsplit('.', 1)[1].lower()

    environ = request.environ
    if file_extension == 'jpg':
        return send_from_directory(directory, latest_file)
    else:
        return send_from_directory(directory, latest_file)

def get_frame():
    folder_path = os.getcwd()
    mp4_files = 'output.mp4'
    video = cv2.VideoCapture(mp4_files)
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)

@app.route("/video_feed")
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/webcam_feed")
def webcam_feed():
    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = 'webcam_output.mp4'
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))

    def generate():
        while True:
            success, frame = cap.read()
            if not success:
                break

            model = YOLO('best.pt')
            results = model(frame, save=True)

            res_plotted = results[0].plot()
            out.write(res_plotted)

            ret, buffer = cv2.imencode('.jpg', res_plotted)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov8 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = YOLO('best.pt')
    app.run(host="0.0.0.0", port=args.port)
