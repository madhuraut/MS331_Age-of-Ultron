from flask_socketio import SocketIO
from flask import Flask, render_template, request
from flask import Blueprint, render_template, request, redirect, flash, url_for

app = Flask(__name__)
socket_io = SocketIO(app)

@app.route('/')
def index():
   #COVER PAGE
    return render_template('base.html')

@app.route('/face')
def face():
    #face RECONGITION TEMPLATE
    return render_template('drishti_indexresent2_base.html')

@socke_tio.on('connect', namespace='/')
def connect_to_web():
    print('[Message] Cover page client connected: {}'.format(request.sid))

@socket_io.on('disconnect', namespace='/')
def disconnect_from_web():
    print('[Message] Cover page client disconnected: {}'.format(request.sid))


@socket_io.on('connect', namespace='/web')
def connect_to_web():
    print('[Message] Face Recognition Web client connected: {}'.format(request.sid))


@socket_io.on('disconnect', namespace='/web')
def disconnect_from_web():
    print('[Message] Face Recognition Web client disconnected: {}'.format(request.sid))


@socket_io.on('connect', namespace='/cv')
def connect_to_cv():
    print('[Message] Computer Vision client connected: {}'.format(request.sid))


@socket_io.on('disconnect', namespace='/cv')
def disconnect_from_cv():
    print('[INFO] Computer Vision client disconnected: {}'.format(request.sid))


@socket_iot.on('cv2server')
def send_messages_(message):
    socketio.emit('server2web', message, namespace='/web')
   


if __name__ == "__main__":
    print('[MESSAGE] The server is starting at http://localhost:5001')
    socketio.run(app=app, host='0.0.0.0', port=5001)
