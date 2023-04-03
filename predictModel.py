import pandas as pd
import pickle
from sklearn import preprocessing
import os
import csv
from chatGPT import runchatGPT
from random import randint

# Load the AI model
model = pickle.load(open('model.pkl', 'rb'))

def dataCleaning(file):
    data = pd.concat([pd.read_csv("data.csv"), pd.read_csv(file)])

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 500)
    for col in data.columns:
        if data[col].dtypes == "bool":
            data[col] = data[col].astype(int)
    return data


def label_encoder(dataframe):
    for col in dataframe.columns:
        if dataframe[col].nunique() < 3 and str(dataframe[col].dtypes) in ["category","object"] :
            le = preprocessing.LabelEncoder()
            dataframe[col] = le.fit_transform(dataframe[col].astype(str))
            return dataframe
        

def one_hot(dataframe):
    for col in dataframe.columns:
        if 2 < dataframe[col].nunique() < 8 and str(dataframe[col].dtypes) in ["category", "object"]:
            dataframe = pd.get_dummies(dataframe, columns=[col])
    return dataframe

def predictModel(file):
    data = dataCleaning(file)
    data = label_encoder(data)
    data = one_hot(data)

    data["person_emp_length"] = data["person_emp_length"].fillna(data["person_emp_length"].mean())
    data["loan_int_rate"] = data["loan_int_rate"].fillna(data["loan_int_rate"].mean())

    y = data.iloc[-1]
    y.drop('loan_status')
    return  model.predict(y)

def getClientsAndPredict(fileName):
	finalData = []
	i = 0
	with open(fileName, 'r', encoding="utf8") as file:
		reader = csv.reader(file)
		for row in reader:
			if i == 0:
				i += 1
			else:
				try:
					csvFile = open('client.csv', 'w+')
					csvData = f'person_age,person_income,person_home_ownership,person_emp_length,loan_intent,loan_grade,loan_amnt,loan_int_rate,loan_status,loan_percent_income,cb_person_default_on_file,cb_person_cred_hist_length\n{",".join([row[j] for j in range(0,12)])}'
					csvFile.writelines(csvData)
					csvFile.close()
					reason = runchatGPT(f'Give Me A Clear And Concise Explanation Why This Person Is Not Eligable To Get A Credit From The Bank According To The Following CSV Data In Less Than 20 Words : {csvData}')
					print(reason)
					finalData.append([i, predictModel('client.csv'), reason, randint(1000, 9999)])
				except Exception as e:
					pass
				i += 1
	os.remove('client.csv')
	return finalData
