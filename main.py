from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('modeloMLPClassifier.pkl')

dados = pd.read_csv('Airlines_Processado.csv')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        # Obter dados do formulário
        data = request.form.to_dict()
        app.logger.info("Dados recebidos: %s", data)

        # Preparar dados para fazer a previsão
        input_data = pd.DataFrame({
            'Airline': [data['Airline']],
            'Flight': [data['Flight']],
            'AirportFrom': [data['AirportFrom']],
            'AirportTo': [data['AirportTo']],
            'DayOfWeek': [data['DayOfWeek']],
            'Time': [data['Time']],
            'Length': [data['Length']]
        })

        # Fazer a previsão
        prediction = model.predict(input_data)[0]
        app.logger.info("Previsão: %s", prediction)

        # Retornar a previsão como JSON
        return jsonify({'prediction': str(prediction)})

    except Exception as e:
        app.logger.error("Erro: %s", str(e))
        return jsonify({'error': str(e)})

@app.route('/verificar', methods=['POST', 'GET'])
def verificar():
    pais = request.form.get('pais-origem')
    p = request.form['pais-origem']
    estado = request.form.get('seu-estado')
    companhia = request.form.get('companhia-aerea')
    voo = request.form.get('numero-voo')

    return f'<p>{pais} {estado} {companhia} {voo}</p>'


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/buscar_voo')
def ir_para_buscar_voo():
    return render_template('buscarvoo.html')


@app.route("/perfil")
def perfil():
    return render_template('perfil.html')


@app.route('/perfil')
def ir_para_perfil():
    return render_template('perfil.html')


@app.route('/cadastro')
def ir_para_cadastro():
    return render_template('cadastro.html')


@app.route('/cadastro')
def cadastro():
    return render_template('cadastro.html')


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/home')
def ir_para_home():
    return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True)


