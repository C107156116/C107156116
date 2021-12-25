import numpy as np
import pickle
import pandas as pd
import json
import heapq
from flask import Flask, jsonify,request
from flask_mysqldb import MySQL
from flask_cors import CORS
from json import dumps
from flask import Flask, make_response
from flask import Response

import sklearn
print(sklearn.__version__)
app = Flask(__name__)
app.config['MYSQL_HOST']='remotemysql.com'
app.config['MYSQL_USER']='GqD8cGeo5O'
app.config['MYSQL_PASSWORD']='We4Vb60cQA'
app.config['MYSQL_DB']='GqD8cGeo5O'
app.config['JSON_AS_ASCII'] = False

mysql=MySQL(app)
df=pd.read_excel('青春露_data_text_process_test_smote.xls')
all_cols=['好吸收','透亮','保濕','不引起過敏','不油膩','溫和低刺激','不致痘','不黏膩','修護','春','夏','秋','冬']
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
    pickle_in = open('product_predict_model\青春露.pickle','rb')
    arr=np.array(process_data)
    print(arr)
    arr=arr.reshape(1,15)
    forest = pickle.load(pickle_in)
    predict_result = forest.predict(arr)
    
    return(str(predict_result[0]))
#    return make_response(dumps(inserValues))
@app.route('/getdata_product1')
def  getdata():
     tmp=[]
     table=[]
     mycursor = mysql.connection.cursor()
     mycursor.execute("SELECT * FROM PRODUCT_1")
     data = mycursor.fetchall()
     for i in range(0,len(data),1):
         for x in range(0,len(data[i]),1):
             tmp.append(data[i][x])
         table.append(tmp)
         tmp=[]
     field_names = [i[0] for i in mycursor.description]
     data=pd.DataFrame(table,columns=field_names)
     return_data=data.to_dict('records')
     json_string = json.dumps(return_data,ensure_ascii = False)
     response = Response(json_string,content_type="application/json; charset=utf-8" )
     return response
@app.route('/getcol_product1')
def  getcol_product1():
     json_string = json.dumps(all_cols,ensure_ascii = False)
     response = Response(json_string,content_type="application/json; charset=utf-8" )
     return response
@app.route('/search_product',methods=['POST'])
def  searchproduct():
     tmp=[]
     table=[]
     inserValues=request.get_json()
     brand=inserValues['brand']
     classfication=inserValues['classfication']


     mycursor = mysql.connection.cursor()
     print(brand)
     if brand=="" and classfication!="":
         mycursor.execute("SELECT * FROM product_list WHERE product_classification=%s",([classfication]))
     elif classfication=="" and brand!="":
         mycursor.execute("SELECT * FROM product_list WHERE product_brand=%s",([brand]))
     elif brand=="" and classfication=="":
         mycursor.execute("SELECT * FROM product_list")
     else:
         mycursor.execute("SELECT * FROM product_list WHERE product_classification=%s AND product_brand=%s",(classfication,brand))
     data = mycursor.fetchall()
     for i in range(0,len(data),1):
         for x in range(0,len(data[i]),1):
             tmp.append(data[i][x])
         table.append(tmp)
         tmp=[]
     field_names = [i[0] for i in mycursor.description]
     data=pd.DataFrame(table,columns=field_names)
     return_data=data.to_dict('records')
     json_string = json.dumps(return_data,ensure_ascii = False)

     if len(data)==0:
         response = "查無資料"
     else:
         response = Response(json_string,content_type="application/json; charset=utf-8" )
         
     return response
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
