from flask import Flask, render_template, request, redirect, flash
import csv
import openai
import tiktoken
import pandas as pd
import pickle
from sklearn import preprocessing

#openai.api_key = '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'
app = Flask(__name__)
Dir = 'data/data.csv'
"""system_message = {"role": "system", "content": "You are a helpful assistant."}
max_response_tokens = 250
token_limit= 4096
conversation=[]
conversation.append(system_message)"""

# Load the AI model
model = pickle.load(open('model.pkl', 'rb'))

data = pd.concat([pd.read_csv("data.csv"), pd.read_csv("test.csv")])

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

for col in data.columns:
    if data[col].dtypes == "bool":
        data[col] = data[col].astype(int)



def label_encoder(dataframe):
    for col in dataframe.columns:
        if dataframe[col].nunique() < 3 and str(dataframe[col].dtypes) in ["category", "object"]:
            le = preprocessing.LabelEncoder()
            dataframe[col] = le.fit_transform(dataframe[col].astype(str))
            return dataframe

def one_hot(dataframe):
    for col in dataframe.columns:
        if 2 < dataframe[col].nunique() < 8 and str(dataframe[col].dtypes) in ["category", "object"]:
            dataframe = pd.get_dummies(dataframe, columns=[col])
    return dataframe

def model(data):
    data = label_encoder(data)
    data = one_hot(data)

    data["person_emp_length"] = data["person_emp_length"].fillna(data["person_emp_length"].mean())
    data["loan_int_rate"] = data["loan_int_rate"].fillna(data["loan_int_rate"].mean())

    y = data.iloc[-1]
    y.drop('loan_status')
    return  model.predict(y)

"""
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def chatGPT(prompt):
    global conversation
    conversation.append({"role": "user", "content": prompt})
    conv_history_tokens = num_tokens_from_messages(conversation)

    while (conv_history_tokens+max_response_tokens >= token_limit):
        del conversation[1]
        conv_history_tokens = num_tokens_from_messages(conversation)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = conversation,
        temperature=1,
        max_tokens=max_response_tokens,
        top_p=0.9
    )

    conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    return str(response['choices'][0]['message']['content'])
"""

def readCSV(fileName):
	data = []
	i = 0
	with open(fileName, 'r', encoding="utf8") as file:
		reader = csv.reader(file)
		for row in reader:
			if i == 0:
				i += 1
			else:
				data.append([row[j] for j in range(0,10)])
				i += 1
	return data
#data = csvToJson(fileName)
#print(readCSV(Dir))

"""
@app.route('/', methods=['POST', 'GET'])
def index():
	if request.method == 'POST':
		file = request.files['data']
		filename = file.filename
		if filename != '' and filename.endswith('.csv'):
			#print(file.read())
			file.save(Dir)
			return 'done'
		else:
			print('File Type Not ALLOWED Or File Not Exist')
			return redirect('/')
	return render_template('index.html', title='index Title')


if __name__ == '__main__':
	app.run(debug=True)
"""

print(model(data))