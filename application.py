from flask import Flask, render_template, url_for, request, jsonify, make_response, flash, redirect
from sklearn.externals import joblib
import librosa

import cough as CP
import text_api

import pandas as pd
import numpy as np
#from flask_cors import CORS
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#def allowed_file(filename):
#    return '.' in filename and \
#           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    dicti = data()
    return render_template('app.html',predic=dicti)
        

@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        try: 
            age = request.form.get('age')
            gender = request.form.get('gender')
            smoker = request.form.get('smoker')
            symptoms = request.form.getlist('reported_symptoms')
            medical_history = request.form.getlist('medical_history')
            symptoms = ",".join(symptoms) + ","
            medical_history = ",".join(medical_history) + ","
            # hasham = request.files
            hasham = request.files.get("cough_data")

            # Textual model
            response = {"age": [int(age)], "gender": [gender],
                "smoker": [smoker], "patient_reported_symptoms": [symptoms],
                "medical_history": [medical_history]
                }
            df1 = pd.DataFrame(response)
            # print(df1)
            prediction = text_api.predict(df1, "./model81.pkl")
            
            # pp = os.getcwd()
            path = "./uploads/hasham.wav"
            
            with open(path, 'wb') as ft:
                ft.write(hasham.read())
            
            # fp = open (path, 'wb')
            # fp.write (hasham.read())
            # fp.close()
            fil  = "./uploads/hasham.wav"    
            f, sr = librosa.load(fil, 22050)
            duration = librosa.get_duration(y=f, sr=sr)
            # print(f)
            print("Duration is: " , duration)
            
            # print(hasham.read())

            # return symptoms
            # return jsonify(hasham.read())
            # check if the post request has the file part
            # if 'file' not in request.files:
            #     flash('No file part')
            #     return redirect(request.url)
            # file = request.files['file']

            # # if user does not select file, browser also
            # # submit an empty part without filename
            # if file.filename == '':
            #     flash('No selected file')
            #     return redirect(request.url)
            # if file and allowed_file(file.filename):
            #     filename = secure_filename(file.filename)
            #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # audio_path = './cough_rec_Int/uploads/'+filename

            cough_result = CP.predict(fil, './cough_model.pkl')

            if prediction[0] == 0 and cough_result == 0:
                return "Hurray! You are safe. You are Covid free!!!"
            elif prediction[0] == 0 and cough_result > 0:
                return "We are worried! You need to visit doctor.!!!"
            elif prediction[0] == 1 and cough_result > 0:
                return "We are worried! You need to visit doctor!!!"
            elif prediction[0] == 1 and cough_result == 0:
                return "Hurray! You are safe. You are Covid free.!!!"

            
            # if cough_result > 0 and textual_model == 0:
            #     return "We are worried! You need to visit doctor"
            # elif cough_result > 0 and textual_model == 1:
            #     return "We are worried! You need to visit doctor"

            # elif cough_result > 0 and textual_model == 1:
            #     return "Hurray! You are safe. You are Covid free!!!"



            # response = {"age": [int(age)], "gender": [gender],
            # "smoker": [smoker], "patient_reported_symptoms": [symptoms],
            # "medical_history": [medical_history]
            # }
            # df1 = pd.DataFrame(response)
            # print(df1)
            # # prediction = text_api.predict(df1, "./cough_testing/model81.pkl")
            # if prediction[0] == 0:
            #     return "Great, you are out of danger according to our model keep following precautions." + audio_path
            # else:
            #     return "You are at risk according to our model, consult a doctor and keep yourself away from others." + audio_path

        except:
            return "Please check if the values are entered correctly"
    
#app.run(debug=True)




