import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Supondo que o dataset tenha colunas 'text' para a avaliação e 'label' para a classificação
data = pd.read_csv('product_reviews.csv')
data = data[['text', 'label']]  # Ajuste os nomes das colunas conforme necessário

# Dividir os dados em treino e teste
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42)

# Transformar os textos em vetores de contagem ou TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
train_vectors = vectorizer.fit_transform(train_texts)
test_vectors = vectorizer.transform(test_texts)

# Treinar o modelo Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(train_vectors, train_labels)

# Fazer previsões e avaliar o modelo
nb_predictions = nb_model.predict(test_vectors)
nb_accuracy = accuracy_score(test_labels, nb_predictions)
print(f"Naive Bayes Accuracy: {nb_accuracy}")
print(classification_report(test_labels, nb_predictions))

# Gerar a matriz de confusão
cm = confusion_matrix(test_labels, nb_predictions)

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão - Naive Bayes')
plt.show()

