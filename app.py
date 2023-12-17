from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Carregar o DataFrame e pré-processar os dados
url = 'https://drive.google.com/uc?id=1jyPDip6yQk3W1odQyl0WT2x2Nj19LAL8'
df = pd.read_csv(url)

columns_to_drop = ['Name', 'ID', 'URI', 'Country', 'Genres', 'Followers', 'URI', 'NumGenres']
df = df.drop(columns=columns_to_drop, errors='ignore')

df = df.dropna()

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

X = df.drop('Gender', axis=1, errors='ignore')
y = df['Gender']

model = RandomForestClassifier()
model.fit(X, y)

# Salvar o modelo
joblib.dump(model, 'seu_modelo.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        age = float(request.form.get('Age'))
        popularity = float(request.form.get('Popularity'))

        # Simplificando o conjunto de recursos apenas com idade e popularidade
        input_data = pd.DataFrame({'Age': [age], 'Popularity': [popularity]})

        prediction_code = model.predict(input_data)[0]
        predicted_gender = le.inverse_transform([prediction_code])[0]

        # Adicione a lógica para prever o artista com base no gênero, se necessário
        # Aqui, atribuímos artistas fictícios com base no gênero
        artist_prediction = "Adele" if predicted_gender == 'female' else "Ed Sheeran"

        return render_template('results.html', gender=predicted_gender, artist=artist_prediction)

    return render_template('index.html', prediction=prediction)

@app.route('/results', methods=['GET'])
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)
