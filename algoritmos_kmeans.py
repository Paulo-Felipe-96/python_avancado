from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# dados aleatórios
dados = {
    'x': [25, 34, 22, 27, 33, 33, 31, 22, 35, 34, 67, 54, 57, 43, 50, 57, 59, 52, 65, 47, 49, 48, 35, 33, 44, 45, 38,
          43, 51, 46],
    'y': [79, 51, 53, 78, 59, 74, 73, 57, 69, 75, 51, 32, 40, 47, 53, 36, 35, 58, 59, 50, 25, 20, 14, 12, 20, 5, 29, 27,
          8, 7]
}

# a classe DataFrame define a variável DF como um dataset com duas colunas
# com os valores de x e y
# head() retorna apenas as 5 primeiras linhas

df = DataFrame(dados, columns=['x', 'y'])
print(df.head())

# KMeans

kmeans = KMeans(n_clusters=2)  # instanciando objeto kmeans
kmeans.fit(df)  # o método fit recebe como param o dataframe e já encontra os centroides

# os centroids são iguais ao numero de clusters, que são iguais ao numero de
# colunas definidas no dataframe (2, x e y)
# os clusters são armazenados em .cluster_centers_
centroides = kmeans.cluster_centers_
print(centroides)

"""
Plotando dispersão de dados e salvando em uma imagem (por limitações de se usar uma IDE).

linha 1 definimos o algoritmo do KMeans
linha 2 declaramos 2 centroides, onde [:, 0] indica que estamos selecionados todas as linhas da coluna 0 (1)
e [:, 1] estamos selecionando todas as linhas da coluna 1 (2)
linha 3 e 4 definimos os labels
linha 5 salvamos uma figura do plot realizado

** Os dados em amarelo correspondem a um centroide enquanto os roxos correspondem ao outro centroide
"""
plt.scatter(df['x'], df['y'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroides[:, 0], centroides[:, 1], c='red', s=50)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('my_plot.png', bbox_inches='tight')
