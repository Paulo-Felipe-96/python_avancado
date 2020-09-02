from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as knnc
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
O KNN verifica através dos vizinhos quais são os que possuem
a menor distância entre eles para tentar prever as classes

"""

# importando dataset do tipo iris
iris = datasets.load_iris()

# definindo os nomes para criar as tabelas próximas

df_iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['target'])
print(df_iris.head())

"""
Dividir os valores em treinamentos e teste. Isso quer dizer que
nosso algoritmo de machile learn aprender a partir dos dados contidos no dataset e
generalizar o que aprendeu, assim dividimos os dados em treinamento e teste.

train_test_split divide de forma automática os dados de treino e teste
"""

x = df_iris.iloc[:, :-1].values  # as entradas
y = df_iris.iloc[:, 4].values  # todas as linhas da última coluna (saída)

# 0.20 indica a divisão dos dados do dataset em 20% para teste e
# 80% para treinamento
X_train, X_test, y_train, y_test, = train_test_split(x, y, test_size=0.20)

"""
Agora passamos pela fase de normalização dos dados.
StandardScaler auxilia nesse processo de maneira mais eficiente
"""
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

"""
Treinando o modelo
KNeighborsClassifier

n_neighors indica que queremos apenas os 5 vizinhos mais próximos
"""
classifier = knnc(n_neighbors=5)
classifier.fit(X_train, y_train)  # aplica a classificação

knnc(algorithm='auto', leaf_size=30, metric='minkowski',
     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
     weights='uniform')

"""
Realizando predições (predicts)
"""
y_pred = classifier.predict(X_test)

"""
Métodos de avaliação do modelo
classification_report, confusion_matrix

Essa etapa é avalia se o modelo se comportou bem ou mal
"""
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# plot da matriz de confusão
matriz_confusao = confusion_matrix(y_test, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=matriz_confusao)
plt.savefig('matriz_confusao.png', bbox_inches='tight')