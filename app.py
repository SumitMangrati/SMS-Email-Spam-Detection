import pickle
import numpy as np
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from flask import Flask, request, jsonify, render_template

nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()


    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))


    return " ".join(y)        

app = Flask(__name__)
## Load the model
model = pickle.load(open('model.pkl' , 'rb'))
tfidf = pickle.load(open('vectorizer.pkl','rb')) 

@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict' , methods = ['POST'])
def predict():
    message = request.form['message']
    transformed_message = transform_text(message)
    transformed_message = [transformed_message]
    vectorized_message = tfidf.transform(transformed_message)
    prediction = model.predict(vectorized_message)
    if prediction == 1:
        prediction = 'Spam'
    else:
        prediction = 'Not Spam'
    
    return render_template('home.html', prediction_text = 'The message is {}'.format(prediction))
    #return render_template('home.html', prediction_text = 'The message is {}'.format(prediction))



if __name__ == '__main__':
    app.run(debug=True)