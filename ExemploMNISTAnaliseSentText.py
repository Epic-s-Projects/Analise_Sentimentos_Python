import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Função para preprocessamento de texto
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    return ' '.join(tokens)

# Carregar os dados
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/yelp.csv"
data = pd.read_csv(url)

# Filtrar apenas 1 e 5 estrelas para simplificar (positivo e negativo)
data = data[data['stars'].isin([1, 5])]
data['label'] = data['stars'].apply(lambda x: 1 if x == 5 else 0)

# Preprocessar o texto
data['cleaned_text'] = data['text'].apply(preprocess_text)

# Dividir os dados em treino e teste
X = data['cleaned_text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vetorização de texto
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Treinar o modelo
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Avaliar o modelo
y_pred = model.predict(X_test_vectorized)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Fazer previsões em novos textos
new_text = ["This product is amazing!", "I did not like this item."]
new_text_cleaned = [preprocess_text(text) for text in new_text]
new_text_vectorized = vectorizer.transform(new_text_cleaned)
predictions = model.predict(new_text_vectorized)
print(predictions)
