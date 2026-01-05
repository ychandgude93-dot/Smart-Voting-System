from flask import Flask, render_template, url_for, request, session, flash, redirect
from flask_wtf import CSRFProtect
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import pymysql
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
import shutil
import datetime
import time
import requests
facedata = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(facedata)

mydb=pymysql.connect(host='localhost', user='root', password='1234', port=3306, database='smart_voting_system')

sender_address = 'ychandgude93@gmail.com'  # enter sender's email id
sender_pass = 'uscl labg xtlz vbvd'  # Gmail app password for OTP sending

app=Flask(__name__)
app.config['SECRET_KEY']='ajsihh98rw3fyes8o3e9ey3w5dc'
csrf = CSRFProtect(app)

@app.before_request
def initialize():
    if 'IsAdmin' not in session:
        session['IsAdmin'] = False
    if 'User' not in session:
        session['User'] = None

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/admin', methods=['POST','GET'])
def admin():
    if request.method=='POST':
        email = request.form['email']
        password = request.form['password']
        if (email=='admin@voting.com') and (password=='admin'):
            session['IsAdmin']=True
            session['User']='admin'
            flash('Admin login successful','success')
    return render_template('admin.html', admin=session['IsAdmin'])

@app.route('/add_nominee', methods=['POST','GET'])
def add_nominee():
    if request.method=='POST':
        member=request.form['member_name']
        party=request.form['party_name']
        logo=request.form['test']
        nominee=pd.read_sql_query('SELECT * FROM nominee', mydb)
        all_members=nominee.member_name.values
        all_parties=nominee.party_name.values
        all_symbols=nominee.symbol_name.values
        if member in all_members:
            flash(r'The member already exists', 'info')
        elif party in all_parties:
            flash(r"The party already exists", 'info')
        elif logo in all_symbols:
            flash(r"The logo is already taken", 'info')
        else:
            sql="INSERT INTO nominee (member_name, party_name, symbol_name) VALUES (%s, %s, %s)"
            cur=mydb.cursor()
            cur.execute(sql, (member, party, logo))
            mydb.commit()
            cur.close()
            flash(r"Successfully registered a new nominee", 'primary')
    return render_template('nominee.html', admin=session['IsAdmin'])

@app.route('/registration', methods=['POST','GET'])
def registration():
    if request.method=='POST':
        import base64
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        state = request.form['state']
        d_name = request.form['d_name']
        middle_name = request.form['middle_name']
        aadhar_id = request.form['aadhar_id']
        voter_id = request.form['voter_id']
        pno = request.form['pno']
        age = int(request.form['age'])
        email = request.form['email']
        img_data = request.form.get('captured_data')
        voters=pd.read_sql_query('SELECT * FROM voters', mydb)
        all_aadhar_ids=voters.aadhar_id.values
        all_voter_ids=voters.voter_id.values
        if age >= 18:
            if (aadhar_id in all_aadhar_ids) or (voter_id in all_voter_ids):
                flash(r'Already Registered as a Voter')
            elif not img_data or not img_data.startswith('data:image'):
                flash('Please capture your photo before submitting registration.', 'danger')
            else:
                # Save image
                path_to_store = os.path.join(os.getcwd(), "all_images", aadhar_id)
                os.makedirs(path_to_store, exist_ok=True)
                img_path = os.path.join(path_to_store, "manual_capture.jpg")
                header, encoded = img_data.split(',', 1)
                with open(img_path, "wb") as f:
                    f.write(base64.b64decode(encoded))
                # Insert user with image path
                sql = 'INSERT INTO voters (first_name, middle_name, last_name, aadhar_id, voter_id, email, pno, state, d_name, verified, image_path) VALUES (%s,%s,%s, %s, %s, %s, %s, %s, %s, %s, %s)'
                cur=mydb.cursor()
                cur.execute(sql, (first_name, middle_name, last_name, aadhar_id, voter_id, email, pno, state, d_name, 'no', img_path))
                mydb.commit()
                cur.close()
                session['aadhar'] = aadhar_id
                session['status'] = 'no'
                session['email'] = email
                flash('Registration complete! Photo captured and saved. Please verify your email.', 'success')
                return redirect(url_for('verify'))
        else:
            flash("if age less than 18 than not eligible for voting","info")
    return render_template('voter_reg.html')

@app.route('/verify', methods=['POST','GET'])
def verify():
    if session['status']=='no':
        if request.method=='POST':
            otp_check=request.form['otp_check']
            if otp_check==session['otp']:
                session['status']='yes'
                sql="UPDATE voters SET verified='%s' WHERE aadhar_id='%s'"%(session['status'], session['aadhar'])
                cur=mydb.cursor()
                cur.execute(sql)
                mydb.commit()
                cur.close()
                flash(r"Email verified successfully",'primary')
                return redirect(url_for('capture_images')) #change it to capture photos
            else:
                flash(r"Wrong OTP. Please try again.","info")
                return redirect(url_for('verify'))
        else:
            #Sending OTP
            message = MIMEMultipart()
            receiver_address = session['email']
            message['From'] = sender_address
            message['To'] = receiver_address
            Otp = str(np.random.randint(100000, 999999))
            session['otp']=Otp
            message.attach(MIMEText(session['otp'], 'plain'))
            abc = smtplib.SMTP('smtp.gmail.com', 587)
            abc.starttls()
            abc.login(sender_address, sender_pass)
            text = message.as_string()
            abc.sendmail(sender_address, receiver_address, text)
            abc.quit()
    else:
        flash(r"Your email is already verified", 'warning')
    return render_template('verify.html')

@app.route('/capture_images', methods=['POST','GET'])
def capture_images():
    if request.method=='POST':
        import base64
        aadhar = session.get('aadhar')
        img_data = request.form.get('captured_data')
        path_to_store = os.path.join(os.getcwd(), "all_images", aadhar)
        os.makedirs(path_to_store, exist_ok=True)
        img_path = os.path.join(path_to_store, "manual_capture.jpg")
        if img_data and img_data.startswith('data:image'):
            header, encoded = img_data.split(',', 1)
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(encoded))
            # If registration is pending, insert user now
            pending = session.pop('pending_registration', None)
            if pending:
                sql = 'INSERT INTO voters (first_name, middle_name, last_name, aadhar_id, voter_id, email,pno,state,d_name, verified, image_path) VALUES (%s,%s,%s, %s, %s, %s, %s, %s, %s, %s, %s)'
                cur=mydb.cursor()
                cur.execute(sql, (pending['first_name'], pending['middle_name'], pending['last_name'], pending['aadhar_id'], pending['voter_id'], pending['email'], pending['pno'], pending['state'], pending['d_name'], pending['verified'], img_path))
                mydb.commit()
                cur.close()
                flash("Registration complete! Photo captured and saved.","success")
                return redirect(url_for('verify'))
            # Optionally update DB with image path for other flows
            cur = mydb.cursor()
            cur.execute("UPDATE voters SET image_path=%s WHERE aadhar_id=%s", (img_path, aadhar))
            mydb.commit()
            cur.close()
            flash("Photo captured and saved!","success")
        else:
            flash("No image data received.","danger")
        return redirect(url_for('home'))
    return render_template('capture.html')

from sklearn.preprocessing import LabelEncoder
import pickle
le = LabelEncoder()
TRAIN_IMAGES_DIR = r"C:\Users\Yash\Pictures\Camera Roll"

def getImagesAndLabels(path):
    faces = []
    Ids = []
    global le
    if not os.path.exists(path):
        return faces, []

    entries = [os.path.join(path, f) for f in os.listdir(path)]
    # if the folder contains subfolders (each person in its own folder)
    if entries and os.path.isdir(entries[0]):
        for folder in entries:
            imagePaths = [os.path.join(folder, f) for f in os.listdir(folder)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            label = os.path.basename(folder)
            for imagePath in imagePaths:
                try:
                    pilImage = Image.open(imagePath).convert('L')
                except Exception:
                    continue
                imageNp = np.array(pilImage, 'uint8')
                faces.append(imageNp)
                Ids.append(label)
    else:
        # folder contains images directly; use the folder name as a single label
        imagePaths = [p for p in entries if os.path.isfile(p) and p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        label = os.path.basename(path)
        for imagePath in imagePaths:
            try:
                pilImage = Image.open(imagePath).convert('L')
            except Exception:
                continue
            imageNp = np.array(pilImage, 'uint8')
            faces.append(imageNp)
            Ids.append(label)

    if not Ids:
        return faces, []
    Ids_new = le.fit_transform(Ids).tolist()
    with open('encoder.pkl', 'wb') as output:
        pickle.dump(le, output)
    return faces, Ids_new

def create_lbph_recognizer():
	"""
	Try multiple ways to create an LBPH recognizer to support different OpenCV builds.
	If none are available, return None.
	"""
	# Preferred: opencv-contrib builds expose cv2.face with LBPHFaceRecognizer_create
	try:
		if hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create"):
			return cv2.face.LBPHFaceRecognizer_create()
	# Older API name
	except Exception:
		pass
	# Fallbacks: older / alternative function names
	if hasattr(cv2, "LBPHFaceRecognizer_create"):
		try:
			return cv2.LBPHFaceRecognizer_create()
		except Exception:
			pass
	if hasattr(cv2, "createLBPHFaceRecognizer"):
		try:
			return cv2.createLBPHFaceRecognizer()
		except Exception:
			pass
	# No supported constructor found
	return None

@app.route('/train', methods=['POST','GET'])
def train():
    if request.method=='POST':
        recognizer = create_lbph_recognizer()
        if recognizer is None:
            flash("LBPH recognizer unavailable. Install opencv-contrib-python (pip install opencv-contrib-python) and restart the app.", "danger")
            return redirect(url_for('train'))
        # use Camera Roll folder as the training images source
        faces, Id = getImagesAndLabels(TRAIN_IMAGES_DIR)
        print(Id)
        print(len(Id))
        recognizer.train(faces, np.array(Id))
        recognizer.save("Trained.yml")
        flash(r"Model Trained Successfully", 'Primary')
        return redirect(url_for('home'))
    return render_template('train.html')
@app.route('/update')
def update():
    return render_template('update.html')
@app.route('/updateback', methods=['POST','GET'])
def updateback():
    if request.method=='POST':
        import base64
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        middle_name = request.form['middle_name']
        aadhar_id = request.form['aadhar_id']
        voter_id = request.form['voter_id']
        email = request.form['email']
        pno = request.form['pno']
        age = int(request.form['age'])
        img_data = request.form.get('captured_data')
        img_path = None
        if img_data and img_data.startswith('data:image'):
            img_dir = os.path.join(os.getcwd(), "all_images", aadhar_id)
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(img_dir, "update.jpg")
            header, encoded = img_data.split(',', 1)
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(encoded))
        voters=pd.read_sql_query('SELECT * FROM voters', mydb)
        all_aadhar_ids=voters.aadhar_id.values
        if age >= 18:
            if (aadhar_id in all_aadhar_ids):
                sql="UPDATE VOTERS SET first_name=%s, middle_name=%s, last_name=%s, voter_id=%s, email=%s,pno=%s, verified=%s, image_path=%s where aadhar_id=%s"
                cur=mydb.cursor()
                cur.execute(sql, (first_name, middle_name, last_name, voter_id, email, pno, 'no', img_path if img_path else '', aadhar_id))
                mydb.commit()
                cur.close()
                session['aadhar']=aadhar_id
                session['status']='no'
                session['email']=email
                flash(r'Database Updated Successfully','Primary')
                return redirect(url_for('verify'))
            else:
                flash(f"Aadhar: {aadhar_id} doesn't exists in the database for updation", 'warning')
        else:
            flash("age should be 18 or greater than 18 is eligible", "info")
    return render_template('update.html')

@app.route('/voting', methods=['POST','GET'])
def voting():
    if request.method=='POST':
        import base64
        # Save the captured image from the browser
        img_data = request.form.get('captured_data')
        img_path = None
        if img_data and img_data.startswith('data:image'):
            img_dir = os.path.join(os.getcwd(), "all_images", "voting")
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(img_dir, f"vote_{int(time.time())}.jpg")
            header, encoded = img_data.split(',', 1)
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(encoded))
        # Now run recognition on the saved image
        encoder_path = os.path.join(os.getcwd(), 'encoder.pkl')
        if not os.path.exists(encoder_path):
            flash('Model encoder not found. Please train the model first.', 'warning')
            return redirect(url_for('train'))
        with open(encoder_path, 'rb') as pkl_file:
            my_le = pickle.load(pkl_file)
        recognizer = create_lbph_recognizer()
        if recognizer is None:
            flash("LBPH recognizer unavailable. Install opencv-contrib-python (pip install opencv-contrib-python) and restart the app.", "danger")
            return redirect(url_for('train'))
        trained_path = os.path.join(os.getcwd(), 'Trained.yml')
        if not os.path.exists(trained_path):
            flash('Trained model not found. Please train the model first.', 'warning')
            return redirect(url_for('train'))
        try:
            recognizer.read(trained_path)
        except Exception:
            try:
                recognizer.load(trained_path)
            except Exception as e:
                flash("Unable to load trained model: %s" % str(e), "danger")
                return redirect(url_for('train'))
        det_aadhar = None
        if img_path:
            im = cv2.imread(img_path)
            if im is not None:
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, 1.2, 5)
                for (x, y, w, h) in faces:
                    Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                    if (conf > 40):
                        det_aadhar = my_le.inverse_transform([Id])[0]
                        break
        if det_aadhar:
            session['select_aadhar'] = det_aadhar
            return redirect(url_for('select_candidate'))
        else:
            flash(r"Unable to detect or recognize person. Contact help desk for manual voting", "info")
            return render_template('voting.html')
    return render_template('voting.html')

@app.route('/select_candidate', methods=['POST','GET'])
def select_candidate():
    #extract all nominees
    aadhar = session['select_aadhar']

    df_nom=pd.read_sql_query('select * from nominee', mydb)
    all_nom=df_nom['symbol_name'].values
    sq = "select * from vote"
    g = pd.read_sql_query(sq, mydb)
    all_adhar = g['aadhar'].values
    if aadhar in all_adhar:
        flash("You already voted", "warning")
        return redirect(url_for('home'))
    else:
        if request.method == 'POST':
            vote = request.form['test']
            session['vote'] = vote
            sql = "INSERT INTO vote (vote, aadhar) VALUES ('%s', '%s')" % (vote, aadhar)
            cur = mydb.cursor()
            cur.execute(sql)
            mydb.commit()
            cur.close()
            s = "select * from voters where aadhar_id='" + aadhar + "'"
            c = pd.read_sql_query(s, mydb)
            pno = str(c.values[0][7])
            name = str(c.values[0][1])
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            url = "https://www.fast2sms.com/dev/bulkV2"

            # message = 'Hi ' + name + ' You voted successfully. Thank you for voting at ' + timeStamp + ' on ' + date + '.'
            no = "9515851969"
            message="helloo hai"
            data1 = {
                "route": "q",
                "message": message,
                "language": "english",
                "flash": 0,
                "numbers": no,
            }

            headers = {
                "authorization": "UwmaiQR5OoA6lSTz93nP0tDxsFEhI7VJrfKkvYjbM2C14Wde8g9lvA2Ghq5VNCjrZ4THWkF1KOwp3Bxd",
                "Content-Type": "application/json"
            }

            response = requests.post(url, headers=headers, json=data1)
            print(response)

            flash(r"Voted Successfully", 'Primary')
            return redirect(url_for('home'))
    return render_template('select_candidate.html', noms=sorted(all_nom))

@app.route('/voting_res')
def voting_res():
    votes = pd.read_sql_query('select * from vote', mydb)
    counts = pd.DataFrame(votes['vote'].value_counts())
    counts.reset_index(inplace=True)
    all_imgs=['1.png','2.png','3.jpg','4.png','5.png','6.png']
    all_freqs=[counts[counts['index']==i].iloc[0,1] if i in counts['index'].values else 0 for i in all_imgs]
    df_nom=pd.read_sql_query('select * from nominee', mydb)
    all_nom=df_nom['symbol_name'].values
    return render_template('voting_res.html', freq=all_freqs, noms=all_nom)

if __name__=='__main__':
    app.run(debug=True)
