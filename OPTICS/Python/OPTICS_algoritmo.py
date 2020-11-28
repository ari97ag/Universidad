import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.preprocessing import normalize, StandardScaler

#Carga de la data
X = pd.read_csv('Mall_Customers.csv',sep=';')

#Limpiando la base de datos
drop_features = ['CustomerID', 'Gender']
X = X.drop(drop_features, axis=1)
X.fillna(method='ffill', inplace=True)

#Base limpita
X.head()

# Estandarizando las variables para que sigan un comportameinto normal [0,1]
estandarizado_normal = StandardScaler()
X_estandarizado_normal = estandarizado_normal.fit_transform(X)

# Escalado a [0,1] en las observaciones normalizadas
X_escalado = normalize(X_estandarizado_normal)

X_escalado = pd.DataFrame(X_escalado)
X_escalado.columns = X.columns
X_escalado.head()

# Construcción del modelo de clasificación OPTICS, diferentes metricas

# min_samples: minimo de observaciones en el nucleo de los clusters
# metric: medida de distancia, si no se especifica ocupa << minkowski >>
# xi: La inclinación mínima en el valle, para que sea efectivamnete considerado como un cluster
# min_cluster_size: Número mínimo de observaciones en un cluster. Si no hay ninguna, se utiliza el valor de min_samples.
# Las ultimas dos solo se pueden utilizar cuando cluster_method='xi'.

modelo_optics1 = OPTICS(min_samples=10, metric='manhattan',xi=0.05,
                        min_cluster_size=0.05, cluster_method='xi')
modelo_optics2 = OPTICS(min_samples=10, metric='euclidean',xi=0.05,
                        min_cluster_size=0.05, cluster_method='xi')

# Insertando la base en los modelos
modelo_optics1.fit(X_escalado)
modelo_optics2.fit(X_escalado)

# Clusters según DBSCAN con epsilon = 0.5
DBSCAN1 = cluster_optics_dbscan(reachability = modelo_optics1.reachability_,
                                core_distances = modelo_optics1.core_distances_,
                                ordering = modelo_optics1.ordering_, eps = 0.5)
print(DBSCAN1)

# Clusters según DBSCAN con epsilon = 2.0
DBSCAN2 = cluster_optics_dbscan(reachability = modelo_optics1.reachability_,
                                core_distances = modelo_optics1.core_distances_,
                                ordering = modelo_optics1.ordering_, eps = 2)
print(DBSCAN2)

# Vector con posiciones del 0 al 199
vector = np.arange(len(X_escalado))
print(vector)

# Distancia de alcance en cada punto observado
reachability = modelo_optics1.reachability_[modelo_optics1.ordering_]
print(reachability)

# Observaciones que logran ser clasificadas con exito y aquellas que no
clusters = modelo_optics1.labels_[modelo_optics1.ordering_]
print(clusters)

from sklearn.metrics import silhouette_score
silo=silhouette_score(X, clusters)
print('El valor de la silueta es:',silo)

# Definiendo la ventana de visualizacion
plt.figure(figsize=(10, 7))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])

# Grafica distancia de alcance
colors = ['c.', 'b.', 'r.', 'y.', 'g.']
for Class, colour in zip(range(0, 5), colors):
    Xk = vector[clusters == Class]
    Rk = reachability[clusters == Class]
    ax1.plot(Xk, Rk, colour, alpha=0.3)
ax1.plot(vector[clusters == -1], reachability[clusters == -1], 'k.', alpha=0.3)
ax1.plot(vector, np.full_like(vector, 2., dtype=float), 'k-', alpha=0.5)
ax1.plot(vector, np.full_like(vector, 0.5, dtype=float), 'k-.', alpha=0.5)
ax1.set_ylabel('Distancia $\epsilon$')
ax1.set_title('Gráfica de Alcance')

# OPTICS Clustering
colors = ['c.', 'b.', 'r.', 'y.', 'g.']
for Class, colour in zip(range(0, 5), colors):
    Xk = X_escalado[modelo_optics1.labels_ == Class]
    ax2.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], colour, alpha=0.3)

ax2.plot(X_escalado.iloc[modelo_optics1.labels_ == -1, 0],
         X_escalado.iloc[modelo_optics1.labels_ == -1, 1],
         'k+', alpha=0.1)
ax2.set_title('OPTICS 1 Clustering')

# DBSCAN Clustering con epsilon = 0.5
colors = ['c', 'b', 'r', 'y', 'g', 'greenyellow']
for Class, colour in zip(range(0, 6), colors):
    Xk = X_escalado[DBSCAN1 == Class]
    ax3.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], colour, alpha=0.3, marker='.')

ax3.plot(X_escalado.iloc[DBSCAN1 == -1, 0],
         X_escalado.iloc[DBSCAN1 == -1, 1],
         'k+', alpha=0.1)
ax3.set_title('DBSCAN Clustering $\epsilon = 0,5$')

# DBSCAN Clustering con epsilon = 2.0
colors = ['c.', 'y.', 'm.', 'g.']
for Class, colour in zip(range(0, 4), colors):
    Xk = X_escalado.iloc[DBSCAN2 == Class]
    ax4.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], colour, alpha=0.3)

ax4.plot(X_escalado.iloc[DBSCAN2 == -1, 0],
         X_escalado.iloc[DBSCAN2 == -1, 1],
         'k+', alpha=0.1)
ax4.set_title('DBSCAN Clustering $\epsilon = 2,0$')

plt.tight_layout()
plt.show()