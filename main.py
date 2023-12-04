# from flask import Flask, render_template, request, redirect, url_for
# from joblib import dump, load
# from sklearn import svm
#
# # clf = svm.SVC()
# #
# # dump(clf, 'model/modelo_rede_neural.pkl')
# # clf = load('model/modelo_rede_neural.pkl')
#
# t = [1, 2, 3, 4, 5, 6]
# c = ['a', 'b', 'c', 'd', 'e', 'f']
#
# app = Flask(__name__)
#
# @app.route('/verificar', methods=['POST', 'GET'])
# def verificar():
#     pais = request.form.get('pais-origem')
#     p = request.form['pais-origem']
#     estado = request.form.get('seu-estado')
#     companhia = request.form.get('companhia-aerea')
#     voo = request.form.get('numero-voo')
#
#     return f'<p>{pais} {estado} {companhia} {voo}</p>'
#
#
# @app.route("/")
# def index():
#     return render_template('index.html')
#
#
# # @app.route("/buscar_voo")
# # def buscar_voo():
# #     return render_template('buscarvoo.html')
#
#
# @app.route('/buscar_voo')
# def ir_para_buscar_voo():
#     return render_template('buscarvoo.html')
#
#
# @app.route("/perfil")
# def perfil():
#     return render_template('perfil.html')
#
#
# @app.route('/perfil')
# def ir_para_perfil():
#     return render_template('perfil.html')
#
#
# @app.route('/cadastro')
# def ir_para_cadastro():
#     return render_template('cadastro.html')
#
#
# @app.route('/cadastro')
# def cadastro():
#     return render_template('cadastro.html')
#
#
# @app.route('/home')
# def home():
#     return render_template('home.html')
#
#
# @app.route('/home')
# def ir_para_home():
#     return render_template('home.html')
#
#
# if __name__ == "__main__":
#     app.run(debug=True)

import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Configurações iniciais
seed = 5
np.random.seed(seed)
pd.set_option('display.max_columns', 25)

# Inicializar o aplicativo Flask
model = joblib.load('model/modelo_rede_neural.pkl')
app = Flask(__name__)

# Carregar e preparar os dados (ajustar o caminho do arquivo conforme necessário)
df = pd.read_pickle('model/modelo_rede_neural.pkl')
# df = df.drop(['Date'], axis=1)
# df2 = df.astype('int')

# Preparar dados para o modelo
# X = df[['Time', 'Length', 'Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek']]
# y = df['Delay']



# Supondo que 'sua_coluna' seja a coluna que você está tentando converter
# df['Airline'] = pd.to_numeric(df['Airline'], errors='coerce')
#
# # Agora, você pode lidar com os valores NaN ou prosseguir com a conversão
# df['Airline'] = df['Airline'].fillna(0).astype(int)

mlp = MLPClassifier()
# mlp.fit(X, y)

# Endpoint para fazer previsões

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()
    X_pred = pd.DataFrame([data])
    y_pred = mlp.predict(X_pred)
    return jsonify(predictions=y_pred.tolist())

    time = request.args['Time']

    # length = request.args['Length']
    # airline = request.args['Airline']
    # airportFrom = request.args['AirportFrom']
    # airportTo = request.args['AirportTo']
    # dayOfWeek = request.args['DayOfWeek']

    # print(time, length, airline, airportFrom, airportTo, dayOfWeek)
    print(time)

    return '<p>ois</p>'


# Endpoint para obter a acurácia do modelo
@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    y_pred = mlp.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return jsonify(accuracy=accuracy)

# Rodar o aplicativo
if __name__ == '__main__':
    app.run(port=8085, host='0.0.0.0', debug=True)