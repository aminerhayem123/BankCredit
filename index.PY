from flask import Flask, render_template, request, redirect, flash
from predictModel import getClientsAndPredict
#print(predictModel('test.csv'))
#print(runchatGPT('hello'))
app = Flask(__name__)
Dir = 'data/data.csv'

@app.route('/', methods=['POST', 'GET'])
def index():
	if request.method == 'POST':
		file = request.files['data']
		filename = file.filename
		if filename != '' and filename.endswith('.csv'):
			file.save(Dir)
			predicted = getClientsAndPredict(Dir)
			accepted = 0
			refused = 0
			for _ in predicted:
				if _[1] == 1:
					accepted += 1
				else:
					refused += 1
			return render_template('index.html', title='Credit Prediction', predicted=predicted, enableProgressBar=1, stat=[accepted, refused])
		else:
			print('File Type Not ALLOWED Or File Not Exist')
			return redirect('/')
	return render_template('index.html', title='Credit Prediction', predicted=[], enableProgressBar=0, stat=[])


if __name__ == '__main__':
	app.run(debug=True)