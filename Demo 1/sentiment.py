import pickle, sys
from sklearn.feature_extraction.text import CountVectorizer

# Get the text to analyze
if len(sys.argv) > 1:
    text = sys.argv[1]
else:
    text = input('Text to analyze: ')

# Load the model and the vocabulary and create a CountVectorizer
model = pickle.load(open('sentiment.pkl', 'rb'))
vocabulary=pickle.load(open('vocabulary.pkl', 'rb'))
vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english', vocabulary=vocabulary)

# Pass the vectorized text to the model and print the result
vectorized_text = vectorizer.transform([text])
score = model.predict_proba(vectorized_text)[0][1]
print(score)