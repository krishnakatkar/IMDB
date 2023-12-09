# Add these imports at the top of your script
from flask import Flask,render_template, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        # Vectorize the input text
        review_tfidf = tfidf_vectorizer.transform([review])
        prediction = model.predict(review_tfidf.reshape(1, -1))[0]
        return render_template('index.html', prediction=prediction, review=review)

if __name__ == '__main__':
    #app.run(port=int(os.environ.get('PORT', 5000)), debug=True)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
    #app.run(port=5000)
