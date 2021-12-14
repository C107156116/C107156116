import numpy as np
import pickle
import pandas as pd
import json
import heapq
from flask import Flask, jsonify,request
from flask_cors import CORS
from json import dumps
from flask import Flask, make_response
from sklearn import preprocessing




app = Flask(__name__)
#app.config['MYSQL_HOST']='remotemysql.com'
#app.config['MYSQL_USER']='GqD8cGeo5O'
#app.config['MYSQL_PASSWORD']='BKeOFOJ8xs'
#app.config['MYSQL_DB']='GqD8cGeo5O'
#mysql=MySQL(app)
df=pd.read_excel('青春露_data_text_process_test.xls')
all_cols=['好吸收','明亮. 透亮','保濕','不引起過敏','會回購','不油膩','溫和低刺激','不致痘','不黏膩','修護','春','夏','秋','冬']
cols=['skin_types','age']
CORS(app)

@app.route('/predict',methods=['POST'])
def  postInput():
    inserValues=request.get_json()
    dic={'混合性肌膚':0,'敏感性肌膚':1,'乾性肌膚':2,'油性肌膚':3,'普通性肌膚':4,'先天過敏性肌膚':5}
    inserValues['skin_types']=dic[inserValues['skin_types']]
    process_data=[]
    for y in range(0,len(cols),1):   
        max_num=max(df[cols[y]])
        min_num=min(df[cols[y]])
        pro_num=round(((float(inserValues[cols[y]])-min_num)/(max_num-min_num)),16)
        process_data.append(pro_num)
    for i in all_cols:
        process_data.append(int(inserValues[i]))
        print(inserValues[i])
    pickle_in = open('randomforest.pickle','rb')

    forest = pickle.load(pickle_in)

    predict_result = forest.predict([process_data])
    
    return(str(predict_result[0]))
#    return make_response(dumps(inserValues))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
