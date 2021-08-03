from flask import Flask, render_template, request
from sklearn import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

""" Se importan las stopwords en español para hacer el entrenamiento 
    al vectorizador junto con el dataset ocupado, se importa el modelo
    resultante en la fase de entrenamiento, y luego de que el usuario 
    ingrese una respuesta, el vectorizador lo envia convertido al modelo
    y se obtiene una respuesta.

"""


app = Flask(__name__)
#Entrenamiento al vectorizador con palabras en español
final_stopwords_list = stopwords.words('spanish')

#Modelo fake news importado
loaded_model = pickle.load(open('modeloen.pkl', 'rb'))

#Entrenamiento del vectorizer 
dataframe = pd.read_csv('modificado.csv')
X = dataframe['text']
Y = dataframe['label']

tfvect = TfidfVectorizer()
tfvect.fit(X)
X = tfvect.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)





def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction=" Ocurrio un problema! intente denuevo")

if __name__ == '__main__':
    app.run(debug=True)

