from flask import Flask, request, jsonify
import datetime
import numpy as np
import pandas as pd
from joblib import load

#Cargar el modelo
dt=load('modelo_lgb.joblib')


#Generar el servidor (Back-end)
servidorWeb=Flask(__name__)

@servidorWeb.route('/modelo',methods=['POST'])
def modelo():
    #Procesar datos de entrada
    contenido=request.files['']
    test=pd.read_csv(contenido)
    #Utilizar el modelo
    resultado=dt.predict(test.values)
    #Regresar la salida del modelo
    time=resultado[0]*0.01
    dias=round(time/86400)
    horas=round((time%86400)/3600)
    minutos=round(((time%86400)%3600)/60)
    segundos=round(((time%86400)%3600)%60,2)
    return jsonify(days=str(dias),hours=str(horas),minutes=str(minutos),seconds=str(segundos))


if __name__=='__main__':
    servidorWeb.run(debug=False,host='0.0.0.0',port='8088')
