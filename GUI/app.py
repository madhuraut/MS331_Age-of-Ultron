from flask_socketio import SocketIO
from flask import Flask, render_template, request
from flask import Blueprint, render_template, request, redirect, flash, url_for
import sqlite3
import pandas as pd
#from flask_login import login_required, current_user
#from werkzeug.utils import secure_filename
#import os
#from . import app
#from . import db

app = Flask(__name__)
socketio = SocketIO(app)



@app.route('/')
def index():
    """Home page."""
    return render_template('base.html')


@app.route('/face')
def face():
    return render_template('drishti_indexresent2_base.html')

#added for profile
@app.route('/profile')
def profile():

   db_file="D:/video-streamer-master/App.db"
   con=sqlite3.connect(db_file)
   db_df=pd.read_sql_query("SELECT * FROM ALL_DETECTIONS",con)
   db_df.to_csv('database.tsv', index=False)
   db=con.cursor()
   res=db.execute("SELECT ID, CLASS, DETECTED_ON FROM ALL_DETECTIONS")
   return render_template('tables.html', users=res.fetchall())

@app.route('/profile', methods=["GET", "POST"])
def upload_image():
    db_file="D:/video-streamer-master/App.db"
    if request.method == "POST":
        search_text =request.form.get('search_field')
        conn=sqlite3.connect(db_file)
        db=conn.cursor()
        get_rows=db.execute("SELECT ID, CLASS, DETECTED_ON FROM ALL_DETECTIONS WHERE CLASS like  ?  ",(search_text, ))
        
        for r in get_rows.fetchall():
            print(r)
        return render_template("tables.html", users=get_rows.fetchall())

@app.route('/map')
def show_map():
    return render_template("map.html")



@socketio.on('connect', namespace='/')
def connect_web():
    print('[INFO] home page client connected: {}'.format(request.sid))

@socketio.on('disconnect', namespace='/')
def disconnect_web():
    print('[INFO] home page client disconnected: {}'.format(request.sid))


@socketio.on('connect', namespace='/web')
def connect_web():
    print('[INFO] face Web client connected: {}'.format(request.sid))


@socketio.on('disconnect', namespace='/web')
def disconnect_web():
    print('[INFO] face Web client disconnected: {}'.format(request.sid))

###added for profile
@socketio.on('connect', namespace='/profile')
def connect_web():
     print('[INFO] profile Web client disconnected: {}'.format(request.sid))
### added for profile
@socketio.on('disconnect', namespace='/profile')
def disconnect_web():
     print('[INFO] profile Web client disconnected: {}'.format(request.sid))

@socketio.on('connect', namespace='/map')
def connect_web():
    print('[INFO] profile Map client disconnected: {}'.format(request.sid))

@socketio.on('connect', namespace='/cv')
def connect_cv():
    print('[INFO] CV client connected: {}'.format(request.sid))


@socketio.on('disconnect', namespace='/cv')
def disconnect_cv():
    print('[INFO] CV client disconnected: {}'.format(request.sid))


@socketio.on('cv2server')
def handle_cv_message(message):
    socketio.emit('server2web', message, namespace='/web')
    #print(message)


if __name__ == "__main__":
    print('[INFO] Starting server at http://localhost:5001')
    socketio.run(app=app, host='0.0.0.0', port=5001)
