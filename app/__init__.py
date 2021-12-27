# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 16:57:18 2021

@author: SASAD
"""

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

cols=['skin_types','age']
CORS(app)

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
 
@app.route('/read_productcol',methods=['POST'])
def  read_productcol():
#     json_string = json.dumps(all_cols,ensure_ascii = False)
#     response = Response(json_string,content_type="application/json; charset=utf-8" )
     inserValues=request.get_json()
     return inserValues
 #------這個路徑應該是沒用到了---------
def predict_method(effect_cols,productname,inserValues):
#    effect_cols=['深層清潔','改善粉刺','清潔力好','緊緻毛孔','易沖淨','不引起過敏','溫和低刺激','清爽','控油']
    effect_list=[]
    product_effects=(inserValues['product_effects'])
    effect_df=pd.DataFrame(product_effects)
    for i in range(len(effect_df)):
        for effect_col in effect_cols:
            try:
                effect_df[effect_col][i]=int(effect_df[effect_col][i])
                effect_list.append(effect_df[effect_col][i])
                print(effect_df[effect_col][i])
            except:
                pass
    season_cols=['春','夏','秋','冬']
    season_list=[]
    product_seasons=(inserValues['product_seasons'])
    season_df=pd.DataFrame(product_seasons)
    for i in range(len(season_df)):
        for season_col in season_cols:
            try:
                season_df[season_col][i]=int(season_df[season_col][i])
                season_list.append(season_df[season_col][i])
                print(season_df[season_col][i])
            except:
                pass
#    productname='亞馬遜白泥淨緻毛孔面膜'
    df=pd.read_excel('app/smote/'+productname+'_data_text_process_test_smote.xls')
    dic={'混合性肌膚':0,'敏感性肌膚':1,'乾性肌膚':2,'油性肌膚':3,'普通性肌膚':4,'先天過敏性肌膚':5}
#    all_cols=['深層清潔','改善粉刺','清潔力好','緊緻毛孔','易沖淨','不引起過敏','溫和低刺激','清爽','控油','春','夏','秋','冬']
    inserValues['skin_types']=dic[inserValues['skin_types']]
    process_data=[]
    for y in range(0,len(cols),1):   
        max_num=max(df[cols[y]])
        min_num=min(df[cols[y]])
        pro_num=round(((float(inserValues[cols[y]])-min_num)/(max_num-min_num)),16)
        process_data.append(pro_num)
    process_data=process_data+effect_list+season_list
    print(len(process_data))
    pickle_in = open('app/product_predict_model/'+productname+'.pickle','rb')
    arr=np.array(process_data)
    print(arr)
    arr=arr.reshape(1,len(process_data))
    forest = pickle.load(pickle_in)
    predict_result = forest.predict(arr)
    return predict_result
@app.route('/getcol_product1')
def  getcol_product1():
    inserValues=request.get_json()
    return inserValues
 #------這個路徑應該是沒用到了---------
@app.route('/predict_product1',methods=['POST'])
def  predict_product1():
    inserValues=request.get_json()
    effect_cols=['深層清潔','改善粉刺','清潔力好','緊緻毛孔','易沖淨','不引起過敏','溫和低刺激','清爽','控油']
    productname='亞馬遜白泥淨緻毛孔面膜'
    predict_result=predict_method(effect_cols,productname,inserValues)
    return(str(predict_result[0]))
#----------------------------------------------    
@app.route('/predict_product2',methods=['POST'])
def  predict_product2():
    inserValues=request.get_json()
    productname='激光極淨白淡斑精華'
    df=pd.read_excel('app/smote/'+productname+'_data_text_process_test_smote.xls')
    effect_cols=['透亮','好吸收','保濕','改善暗沉','溫和低刺激','好推勻','用量省','不黏膩','淡化斑點']
    predict_result=predict_method(effect_cols,productname,inserValues)
    return(str(predict_result[0]))
@app.route('/predict_product3',methods=['POST'])
def  predict_product3():
    inserValues=request.get_json()
    productname='青春露'   
    effect_cols=['好吸收','透亮','保濕','不引起過敏','不油膩','溫和低刺激','不致痘','不黏膩','修護']
    predict_result=predict_method(effect_cols,productname,inserValues)
    return(str(predict_result[0]))
@app.route('/predict_product4',methods=['POST'])
def  predict_product4():
    inserValues=request.get_json()
    productname='R.N.A.超肌能緊緻活膚霜(輕盈版)'
    effect_cols=['保濕','清爽','不黏膩','好推勻','好吸收','延展度佳','修護','不厚重','彈力','輕盈']		
    predict_result=predict_method(effect_cols,productname,inserValues)
    return(str(predict_result[0]))
@app.route('/predict_product5',methods=['POST'])
def  predict_product5():
    inserValues=request.get_json()
    productname='嘉美艷容露'
    effect_cols=['價格實在','清爽','收斂','舒緩','不致痘','鎮定','不黏膩','不油膩','控油']		
    predict_result=predict_method(effect_cols,productname,inserValues)
    return(str(predict_result[0]))
@app.route('/predict_product6',methods=['POST'])
def  predict_product6():
    inserValues=request.get_json()
    productname='深層卸粧乳'
    effect_cols=['價格實在','溫和低刺激','易沖淨','清爽','清潔力好','不油膩','不致痘','不緊繃','不引起過敏']		
    predict_result=predict_method(effect_cols,productname,inserValues)
    return(str(predict_result[0]))
@app.route('/getdata',methods=['POST'])
def  getproductdata():
    inserValues=request.get_json()
    productnum=int(inserValues['product_num'])
    print(productnum)
    tmp=[]
    table=[]
    mycursor = mysql.connection.cursor()
    if(productnum==1):
        mycursor.execute("SELECT * FROM product_YAMASHIN");
        print('獲取資料成功')
    elif(productnum==2):
        mycursor.execute("SELECT * FROM product_SUPERLIGHT");
        print('獲取資料成功')
    elif(productnum==3):
        mycursor.execute("SELECT * FROM product_skii_youngwater");
        print('獲取資料成功')
    elif(productnum==4):
        mycursor.execute("SELECT * FROM product_skii_RNA");
    elif(productnum==5):
        mycursor.execute("SELECT * FROM product_skii_RNA_light");
    elif(productnum==6):
        mycursor.execute("SELECT * FROM product_HOMEBEATY");
    elif(productnum==7):
        mycursor.execute("SELECT * FROM product_MINI");
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
#----------------------------------------------  
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
