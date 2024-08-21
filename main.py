import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from deep_translator import GoogleTranslator
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Carregar o dataset do link externo
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/yelp.csv'
data = pd.read_csv(url)

# Ajustar o dataset
data = data[['text', 'stars']]  
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

# Função para criar o gráfico da matriz de confusão
def plot_confusion_matrix():
    fig = plt.Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
    ax.set_xlabel('Previsão')
    ax.set_ylabel('Verdadeiro')
    ax.set_title('Matriz de Confusão - Naive Bayes')
    return fig

# Função para criar o gráfico de distribuição das classes
def plot_class_distribution():
    fig = plt.Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    sns.countplot(x='label', data=data, palette='viridis', ax=ax)
    ax.set_xlabel('Classe')
    ax.set_ylabel('Contagem')
    ax.set_title('Distribuição das Classes')
    ax.set_xticklabels(['Negativo', 'Positivo'])
    return fig

# Importância das palavras no modelo Naive Bayes
feature_names = vectorizer.get_feature_names_out()
class_labels = nb_model.classes_
log_prob = nb_model.feature_log_prob_
sorted_features = np.argsort(log_prob[0])
top_n = 20  # Número de palavras mais importantes a exibir
top_words = [feature_names[i] for i in sorted_features[-top_n:]]

# Traduzir as palavras para o português
translator = GoogleTranslator(source='en', target='pt')
translated_words = [translator.translate(word) for word in top_words]

# Função para criar o gráfico de palavras importantes
def plot_top_words():
    fig = plt.Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.barh(range(top_n), log_prob[0][sorted_features[-top_n:]], align='center', color='purple')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(translated_words)
    ax.set_xlabel('Log Probabilidade')
    ax.set_ylabel('Palavra (Traduzida)')
    ax.set_title('Palavras Mais Importantes - Naive Bayes (Traduzidas)')
    ax.invert_yaxis()
    return fig

# Criar a janela principal
root = tk.Tk()
root.title("Análise de Sentimento")

# Criar as abas
tab_control = ttk.Notebook(root)

tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)

tab_control.add(tab1, text='Matriz de Confusão')
tab_control.add(tab2, text='Distribuição das Classes')
tab_control.add(tab3, text='Palavras Importantes')

tab_control.pack(expand=1, fill='both')

# Adicionar os gráficos às abas
canvas1 = FigureCanvasTkAgg(plot_confusion_matrix(), master=tab1)
canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

canvas2 = FigureCanvasTkAgg(plot_class_distribution(), master=tab2)
canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

canvas3 = FigureCanvasTkAgg(plot_top_words(), master=tab3)
canvas3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Iniciar o loop da interface gráfica
root.mainloop()
