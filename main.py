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
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# seed = 5
# np.random.seed(seed)
# pd.set_option('display.max_columns', 25)

# model = joblib.load('model/modelo_rede_neural.pkl')
app = Flask(__name__)

df = pd.read_pickle('model/modelo_rede_neural.pkl')
df2 = df.astype('int')
#
X = df2[['Time', 'Length', 'Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek']]
y = df2['Delay']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=100, batch_size=10)
mlp.fit(X_train, y_train)


@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()
    X_pred = pd.DataFrame([data])
    y_pred = mlp.predict(X_pred)

    time = request.args['Time']
    print(time)
    print(data)

    teste = np.array([[time]])

    classe = df.predict(teste)

    # length = request.args['Length']
    # airline = request.args['Airline']
    # airportFrom = request.args['AirportFrom']
    # airportTo = request.args['AirportTo']
    # dayOfWeek = request.args['DayOfWeek']

    # print(time, length, airline, airportFrom, airportTo, dayOfWeek)


    return '<p>ois</p>'
    # return jsonify(predictions=y_pred.tolist())



# Endpoint para obter a acurácia do modelo
# @app.route('/accuracy', methods=['GET'])
# def get_accuracy():
#     y_pred = mlp.predict(X)
#     accuracy = accuracy_score(y, y_pred)
#     return jsonify(accuracy=accuracy)

# Rodar o aplicativo
if __name__ == '__main__':
    app.run(port=8085, host='0.0.0.0', debug=True)