import pickle
from flask import Flask, request
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
model = pickle.load(open('sentiment.pkl', 'rb'))
vocabulary = pickle.load(open('vocabulary.pkl', 'rb'))
vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english', vocabulary=vocabulary)

@app.route('/analyze', methods=['GET'])
def analyze():
    if "text" in request.args:
        text = request.args.get('text')
    else:
        return "No string to analyze"

    vectorized_text = vectorizer.transform([text])
    score = model.predict_proba(vectorized_text)[0][1]
    return str(score)

if __name__ == '__main__':
    app.run(debug=True, port=8008, host='0.0.0.0')