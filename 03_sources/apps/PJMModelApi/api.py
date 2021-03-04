from flask import Flask, request
from flask_restful import Resource, Api

import os.path
import time
import datetime

import pandas as pd
import numpy as np
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from sklearn.metrics import mean_squared_error


from hdfs import InsecureClient
from hdfs.util import HdfsError

# !pip install cassandra-driver
from cassandra.cluster import Cluster
from cassandra.query import dict_factory

# Initialize Flask
app=Flask(__name__)
api = Api(app)


model_local_path = "/app/model/"
model_hdfs_remote_path = "/user/root/data/pjm/model/"
model_name = "model_arima.pkl"

#------------ Méthodes d'entrainement et évaluation du modèle ---------------
# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    hist_rmse = list()
    
    #Evaluation préliminaire sur un petit échantillon
    test = test[:366]   
    
    for t in range(len(test)):
        #print(arima_order,"  step = ",t)
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
        hist_rmse.append(sqrt(mean_squared_error(test[:len(predictions)], predictions)))
        #print('Predicted=%.3f, Expected=%.3f, RMSE = %.0f' % (yhat, obs, hist_rmse[-1]))
    

    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    train_size = int(len(dataset) * 0.50)
    train, test = dataset[0:train_size], dataset[train_size:]
    best_score, best_cfg = float("inf"), None
    
    for q in q_values:
        for d in d_values:
            for p in p_values:
                try:
                    arima_order = (p,d,q)
                    rmse, model = evaluate_arima_model(dataset, arima_order)
                   
                    if rmse < best_score:
                        best_score, best_cfg = rmse, arima_order
                    print('ARIMA%s RMSE=%.0f  , actual best : %s RMSE: %.0f' % (arima_order,rmse, best_cfg,best_score))
                except:
                    print("ARIMA%s error",arima_order)
                    continue
                
    return best_cfg, best_score

#---------------- Méthode de communication avec CASSANDRA ------------------
def get_data_cassandra():
    #Connexion au cluster Cassandra
    cluster = Cluster(contact_points=['cassandra'],port=9042)
    session = cluster.connect()
    session.default_timeout = 10

    # Récupération des données Cassandra dans un  modèle
    db_name = "pjm"
    columnFamilyName = "estimated_load_hourly_summary"

    # Sélection de la base de données
    session.execute('use ' + db_name)
    
    query = "SELECT * FROM  " + columnFamilyName + " ;"
    #print(query)
    
    data = pd.DataFrame(session.execute(query, timeout=None))
    #df = pd.DataFrame()
    #for row in session.execute(request_select):
    #    df = df.append(pd.DataFrame(row))
    
    #closing Cassandra connection
    session.shutdown()

    return data



@app.route('/')
def home():
    return("""<h1>PJM Prediction Model API</h1>
        <br><br>Lancement de l'évaluation du modèle : url <a href='http://localhost:5002/evaluate' target='_blank'>localhost:5002/evaluate</a>
        <br><br>Lancement d'une prediction : url <a href='http://localhost:5002/predict/period=10' target='_blank'>localhost:5002/predict/period=10</a>
        <br>
        <br>
        <br>
        <br>
    """)



class EvaluateModel(Resource):
    """
    Fonctions de ré-entrainement du modèle avec les dernières données en base
    """
    def get(self):
        
        # Récupération du Dataset pour l'évaluation
        df = get_data_cassandra()
        
        print(df.head())
        X = df['total_estimated_load'].values

        # evaluate parameters (p,d,q)  <=> (AR, I, MA)
        p_values = 7
        d_values = 0
        q_values = 5
        #best_cfg, best_score = evaluate_models(X, p_values, d_values, q_values)
        best_cfg = (p_values,d_values,q_values)
        
        # Entrainement du meilleur modèle
        model = ARIMA(X, order=best_cfg)
        model_fit = model.fit()
        
        # save model
        if not os.path.exists(model_local_path):
               # Création du dossier d'export local qui n'existe pas
               os.makedirs(model_local_path,exist_ok=False)
        
        model_fit.save(model_local_path + model_name)
            
        # Connexion au client HDFS
        client = InsecureClient(url='http://namenode:9870', user='root')
    
        # Création du dossier de stockage des fichiers traités
        if client.status(model_hdfs_remote_path,strict=False) == None:
                client.makedirs(model_hdfs_remote_path)

	# Copie du modèle sur HDFS
        remote_load_path = client.upload(model_hdfs_remote_path, model_local_path + model_name,overwrite=True)
        #print(remote_load_path)

        print(client.list(model_hdfs_remote_path))

	
        return { 'best_cfg': best_cfg , 'status': 'Terminated'}




class Predict(Resource):
    def get(self,period):
                
        print("Period to predict : ",period)
        
        # Connexion au client HDFS
        client = InsecureClient(url='http://namenode:9870', user='root')
        
        # Vérification de la présence du modèle sauvegardé sur HDFS
        if client.status(model_hdfs_remote_path + model_name , strict=False) != None:
            
            # load model
            client.download(model_hdfs_remote_path+model_name, model_local_path, overwrite=True)
            model_fit = ARIMAResults.load(model_local_path + model_name)
     
            # Dataset pour l'évaluation
            df = get_data_cassandra()
            X = df['total_estimated_load'].values
            
            start_index = len(X)
            end_index = start_index + int(period)
            forecast = model_fit.predict(start=start_index, end=end_index)
            
            day = df['date_est_load'][-1]
            print(type(day))
            day += datetime.timedelta(days=1)
            
            res = {}
            for yhat in forecast:
                res[day] = yhat
                day += datetime.timedelta(days=1)
            
            return res
    
    
        return "Service has been stopped"


api.add_resource(EvaluateModel,'/evaluate')
api.add_resource(Predict,'/predict/<string:period>')


if __name__ == "__main__":
    isRunning = False
    currentFileToImport = ""
    app.run(host="0.0.0.0", port=5002,debug=True)
