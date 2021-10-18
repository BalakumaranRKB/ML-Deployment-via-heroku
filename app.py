from flask import Flask,render_template, request , jsonify
from flask_cors import CORS,cross_origin
import pickle



app = Flask(__name__)

@app.route('/',methods = ['GET'])
@cross_origin()
def homePage():
	return render_template("index.html")

@app.route('/predict',methods = ['POST','GET'])
@cross_origin()

def index():
	if request.method == 'POST':
		try:
			residential_land_zoned_for_lots  = float(request.form['residential_land_zoned_for_lots'])
			is_Charles_River = (request.form['Charles_River'])
			if (is_Charles_River == 'yes'):
				Charles_River=1
			else:
				Charles_River=0
			NOX_concentration  = float(request.form['NOX_concentration'])
			avg_no_of_rooms_per_dwelling  = float(request.form['avg_no_of_rooms_per_dwelling'])
			Dis_to_Employment_Centre  = float(request.form['Dis_to_Employment_Centre'])
			pupil_teacher_ratio  = float(request.form['pupil_teacher_ratio'])
			proportion_of_blacks  = float(request.form['proportion_of_blacks'])
			lower_status_of_population  = float(request.form['lower_status_of_population'])
			crime_rate  = float(request.form['crime_rate'])


			#load the scaler
			filename_1 = 'scaler.pickle'
			scaler_model = pickle.load(open(filename_1,'rb'))

			#transform the test data
			X_test_scaled = scaler_model.transform([[residential_land_zoned_for_lots,Charles_River,NOX_concentration,avg_no_of_rooms_per_dwelling,Dis_to_Employment_Centre,pupil_teacher_ratio,proportion_of_blacks,lower_status_of_population,crime_rate]])
			print(X_test_scaled)

			#load the model
			filename_2  = 'finalized_model.pickle'
			loaded_model = pickle.load(open(filename_2, 'rb')) #loading the model file from the storage

			#make predictions on the test set
			prediction = loaded_model.predict(X_test_scaled)
			print('prediction is', prediction)
			#Showing the prediction results in the UI
			return render_template('results.html',prediction=round(prediction[0]))
		except Exception as e:
			print('The exception message is :',e)
			return'something is wrong'
	else:
		return render_template('index.html')

if __name__ == "__main__":
	app.run(debug = True)

