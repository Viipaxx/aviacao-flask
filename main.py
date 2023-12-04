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

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Configurações iniciais
seed = 5
np.random.seed(seed)
pd.set_option('display.max_columns', 25)

# Inicializar o aplicativo Flask
app = Flask(__name__)

# Carregar e preparar os dados (ajustar o caminho do arquivo conforme necessário)
df = pd.read_csv('Airlines_Processado.csv')
# df = df.drop(['Date'], axis=1)
# df2 = df.astype('int')

# Preparar dados para o modelo

le = LabelEncoder()

df['Airline'] = le.fit_transform(df['Airline'])
df['AirportFrom'] = le.fit_transform(df['AirportFrom'])
df['AirportTo'] = le.fit_transform(df['AirportTo'])
df['DayOfWeek'] = le.fit_transform(df['DayOfWeek'])
df['Delay'] = le.fit_transform(df['Delay'])

X = df[['Time', 'Length', 'Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek']]
y = df['Delay']

# Treinar o modelo de Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X, y)

# Endpoint para fazer previsões
@app.route('/predict', methods=['POST'])
def predict():
    # data = request.get_json()
    # X_pred = pd.DataFrame([data])
    # y_pred = random_forest.predict(X_pred)
    # return jsonify(predictions=y_pred.tolist())

    time = request.form['Time']
    print(time)


# Endpoint para obter a acurácia do modelo
@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    y_pred = random_forest.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return jsonify(accuracy=accuracy)

# Rodar o aplicativo
if __name__ == '__main__':
    app.run()