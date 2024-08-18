import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from deep_translator import GoogleTranslator  # Importar o tradutor

# Carregar o dataset do link externo
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/yelp.csv'
data = pd.read_csv(url)

# Verificar as primeiras linhas do dataset para ajustar os nomes das colunas
print(data.head())

# Supondo que o dataset tenha colunas 'text' para a avaliação e 'label' para a classificação
data = data[['text', 'stars']]  # Ajuste os nomes das colunas conforme necessário

# Mapear estrelas para categorias binárias: 1 (negativo) e 2 (positivo)
data['label'] = data['stars'].apply(lambda x: 1 if x < 3 else 2)

# Dividir os dados em treino e teste
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42)

# Transformar os textos em vetores de TF-IDF
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

# Importância das palavras no modelo Naive Bayes
feature_names = vectorizer.get_feature_names_out()
class_labels = nb_model.classes_

# As probabilidades logarítmicas para cada classe
log_prob = nb_model.feature_log_prob_

# Ordenar as palavras de acordo com sua importância
sorted_features = np.argsort(log_prob[0])

top_n = 20  # Número de palavras mais importantes a exibir

# Obter as palavras mais importantes
top_words = [feature_names[i] for i in sorted_features[-top_n:]]  # Certifique-se de que esta linha exista e funcione

# Traduzir as palavras para o português usando deep-translator
translator = GoogleTranslator(source='en', target='pt')
translated_words = [translator.translate(word) for word in top_words]

# Plotar o gráfico com as palavras traduzidas
plt.figure(figsize=(10, 6))
plt.barh(range(top_n), log_prob[0][sorted_features[-top_n:]], align='center', color='purple')
plt.yticks(range(top_n), translated_words)
plt.xlabel('Log Probabilidade')
plt.ylabel('Palavra (Traduzida)')
plt.title('Palavras Mais Importantes - Naive Bayes (Traduzidas)')
plt.gca().invert_yaxis()
plt.show()
