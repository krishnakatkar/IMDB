{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "a66967f5-2a9e-43eb-8c96-6f81df097e68",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting flask\n",
      "  Downloading flask-3.0.0-py3-none-any.whl.metadata (3.6 kB)\n",
      "Requirement already satisfied: Werkzeug>=3.0.0 in c:\\users\\kkatk\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from flask) (3.0.1)\n",
      "Requirement already satisfied: Jinja2>=3.1.2 in c:\\users\\kkatk\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from flask) (3.1.2)\n",
      "Collecting itsdangerous>=2.1.2 (from flask)\n",
      "  Downloading itsdangerous-2.1.2-py3-none-any.whl (15 kB)\n",
      "Collecting click>=8.1.3 (from flask)\n",
      "  Downloading click-8.1.7-py3-none-any.whl.metadata (3.0 kB)\n",
      "Collecting blinker>=1.6.2 (from flask)\n",
      "  Downloading blinker-1.7.0-py3-none-any.whl.metadata (1.9 kB)\n",
      "Requirement already satisfied: colorama in c:\\users\\kkatk\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from click>=8.1.3->flask) (0.4.6)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\kkatk\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from Jinja2>=3.1.2->flask) (2.1.3)\n",
      "Downloading flask-3.0.0-py3-none-any.whl (99 kB)\n",
      "   ---------------------------------------- 0.0/99.7 kB ? eta -:--:--\n",
      "   ---------------- ----------------------- 41.0/99.7 kB 1.9 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 99.7/99.7 kB 1.9 MB/s eta 0:00:00\n",
      "Downloading blinker-1.7.0-py3-none-any.whl (13 kB)\n",
      "Downloading click-8.1.7-py3-none-any.whl (97 kB)\n",
      "   ---------------------------------------- 0.0/97.9 kB ? eta -:--:--\n",
      "   ---------------------------------------- 97.9/97.9 kB 5.8 MB/s eta 0:00:00\n",
      "Installing collected packages: itsdangerous, click, blinker, flask\n",
      "Successfully installed blinker-1.7.0 click-8.1.7 flask-3.0.0 itsdangerous-2.1.2\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install flask"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "0d2a25fd-6f02-49f3-b5ba-a71dce58e1d0",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "import pickle\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "f0d596b4-d407-4705-82c0-ad188120cec7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "3b649da4-b337-48a6-acf3-1f160c288569",
   "metadata": {},
   "outputs": [],
   "source": [
    "app = Flask(__name__)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "ad0bc910-93f4-4fa8-9b74-f99d627003c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Loading the model which was trained in IMDB dataset traing and the vectorizer which was saved\n",
    "with open('sentiment_model.pkl', 'rb') as model_file:\n",
    "    model = pickle.load(model_file)\n",
    "\n",
    "with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:\n",
    "    tfidf_vectorizer = pickle.load(vectorizer_file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "fa1270aa-36d4-4711-866b-948c3f6058ac",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on http://127.0.0.1:5000\n",
      "Press CTRL+C to quit\n",
      " * Restarting with stat\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[1;31mSystemExit\u001b[0m\u001b[1;31m:\u001b[0m 1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\kkatk\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\IPython\\core\\interactiveshell.py:3556: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.\n",
      "  warn(\"To exit: use 'exit', 'quit', or Ctrl-D.\", stacklevel=1)\n"
     ]
    }
   ],
   "source": [
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    data = request.get_json(force=True)\n",
    "    reviews = data['reviews']\n",
    "\n",
    "    # Vectorizing the input data with the same vector used during model training\n",
    "    reviews_vectorized = tfidf_vectorizer.transform(reviews)\n",
    "\n",
    "    # making the predictions with the trained model\n",
    "    predictions = model.predict(reviews_vectorized)\n",
    "\n",
    "    # Returning the predictions in a json instance\n",
    "    return jsonify({'predictions': predictions.tolist()})\n",
    "if __name__ == '__main__':\n",
    "    app.run(port=5000, debug=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "03dd5ac6-899c-43fb-8bf7-de9fbeebfc68",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
