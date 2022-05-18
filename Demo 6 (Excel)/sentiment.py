import pickle, os
import xlwings as xw
from sklearn.feature_extraction.text import CountVectorizer

# Load the model and the vocabulary and create a CountVectorizer
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sentiment.pkl'))
vocab_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'vocabulary.pkl'))

model = pickle.load(open(model_path, 'rb'))
vocabulary = pickle.load(open(vocab_path, 'rb'))
vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english', vocabulary=vocabulary)

@xw.func
def analyze_text(text):
	vectorized_text = vectorizer.transform([text])
	score = model.predict_proba(vectorized_text)[0][1]
	return score