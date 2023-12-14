
from flask import Flask,render_template, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = Flask(__name__)

# Loading the trained model and TF-IDF vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_review', methods=['POST'])
def predict_review():
    if request.method == 'POST':
        user_review = request.form['user_review']
        # Vectorizing the input text
        review_tfidf = tfidf_vectorizer.transform([user_review])
        review_prediction = model.predict(review_tfidf.reshape(1, -1))[0]
        return render_template('index.html', review_prediction=review_prediction, user_review=user_review)

if __name__ == '__main__':
#app.run(port=int(os.environ.get('PORT', 5000)), debug=True)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
    #app.run(port=5000)
