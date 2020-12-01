import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.preprocessing import normalize, StandardScaler
from sklearn import metrics

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

# Vector con posiciones del 0 al 199
vector = np.arange(len(X_escalado))
print(vector)

# Construcción del modelo de clasificación OPTICS, diferentes metricas
# min_samples: minimo de observaciones en el nucleo de los clusters
# metric: medida de distancia, si no se especifica ocupa << minkowski >>
# xi: La inclinación mínima en el valle, para que sea efectivamnete considerado como un cluster
# min_cluster_size: Número mínimo de observaciones en un cluster. Si no hay ninguna, se utiliza el valor de min_samples.
# Las ultimas dos solo se pueden utilizar cuando cluster_method='xi'.

# Caso distancia manhattan
modelo_optics1 = OPTICS(min_samples=10, metric='manhattan',xi=0.05,
                        min_cluster_size=0.05, cluster_method='xi')

# Insertando la base en los modelos
modelo_optics1.fit(X_escalado)

# Clusters según DBSCAN con epsilon = 0.5
DBSCAN11 = cluster_optics_dbscan(reachability = modelo_optics1.reachability_,
                                core_distances = modelo_optics1.core_distances_,
                                ordering = modelo_optics1.ordering_, eps = 0.5)
print(DBSCAN11)

# Clusters según DBSCAN con epsilon = 2.0
DBSCAN12 = cluster_optics_dbscan(reachability = modelo_optics1.reachability_,
                                core_distances = modelo_optics1.core_distances_,
                                ordering = modelo_optics1.ordering_, eps = 2)
print(DBSCAN12)

# Distancia de accesibilidad en cada punto observado
reachability1 = modelo_optics1.reachability_[modelo_optics1.ordering_]
print(reachability1)

# Clasificación
clusters1 = modelo_optics1.labels_[modelo_optics1.ordering_]
print(clusters1)

# Validacion interna
print("Coeficiente Silhouette: %0.3f"
      % metrics.silhouette_score(X_escalado, clusters1))
# Los clusters estan muy cerca al hecho de estar uno sobre otro
print("Coeficiente Davies-Bouldin: %0.3f"
      % metrics.davies_bouldin_score(X_escalado, clusters1))
# Validacion externa
# La separacion entre los cluster no es muy buena pues tienen un comportameinto jerarquico
print("Coeficiente ARI: %0.3f"
      % metrics.cluster.adjusted_rand_score(clusters1, DBSCAN11))

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
    Xk = vector[clusters1 == Class]
    Rk = reachability1[clusters1 == Class]
    ax1.plot(Xk, Rk, colour, alpha=0.3)
ax1.plot(vector[clusters1 == -1], reachability1[clusters1 == -1], 'k.', alpha=0.3)
ax1.plot(vector, np.full_like(vector, 2., dtype=float), 'k-', alpha=0.5)
ax1.plot(vector, np.full_like(vector, 0.5, dtype=float), 'k-.', alpha=0.5)
ax1.set_ylabel('Puntaje de Accecibilidad')
ax1.set_title('Gráfica de Accesibilidad')

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
    Xk = X_escalado[DBSCAN11 == Class]
    ax3.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], colour, alpha=0.3, marker='.')

ax3.plot(X_escalado.iloc[DBSCAN11 == -1, 0],
         X_escalado.iloc[DBSCAN11 == -1, 1],
         'k+', alpha=0.1)
ax3.set_title('DBSCAN Clustering $\epsilon = 0,5$')

# DBSCAN Clustering con epsilon = 2.0
colors = ['c.', 'y.', 'm.', 'g.']
for Class, colour in zip(range(0, 4), colors):
    Xk = X_escalado.iloc[DBSCAN12 == Class]
    ax4.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], colour, alpha=0.3)

ax4.plot(X_escalado.iloc[DBSCAN12 == -1, 0],
         X_escalado.iloc[DBSCAN12 == -1, 1],
         'k+', alpha=0.1)
ax4.set_title('DBSCAN Clustering $\epsilon = 2,0$')

plt.tight_layout()
plt.show()

###############################
# Caso distancia euclidiana
modelo_optics2 = OPTICS(min_samples=10, metric='euclidean',xi=0.05,
                        min_cluster_size=0.05, cluster_method='xi')

# Insertando la base en los modelos
modelo_optics2.fit(X_escalado)

# Clusters según DBSCAN con epsilon = 0.5
DBSCAN21 = cluster_optics_dbscan(reachability = modelo_optics2.reachability_,
                                core_distances = modelo_optics2.core_distances_,
                                ordering = modelo_optics2.ordering_, eps = 0.5)
print(DBSCAN21)

# Clusters según DBSCAN con epsilon = 2.0
DBSCAN22 = cluster_optics_dbscan(reachability = modelo_optics2.reachability_,
                                core_distances = modelo_optics2.core_distances_,
                                ordering = modelo_optics2.ordering_, eps = 2)
print(DBSCAN22)

# Distancia de accesibilidad en cada punto observado
reachability2 = modelo_optics2.reachability_[modelo_optics2.ordering_]
print(reachability2)

# Clasificación
clusters2 = modelo_optics2.labels_[modelo_optics2.ordering_]
print(clusters2)

# Validacion interna
print("Coeficiente Silhouette: %0.3f"
      % metrics.silhouette_score(X_escalado, clusters2))
# Los clusters estan muy cerca al hecho de estar uno sobre otro
print("Coeficinete Davies-Bouldin: %0.3f"
      % metrics.davies_bouldin_score(X_escalado, clusters2))
# La separacion entre los cluster no es muy buena pues tienen un comportameinto jerarquico

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
    Xk = vector[clusters2 == Class]
    Rk = reachability2[clusters2 == Class]
    ax1.plot(Xk, Rk, colour, alpha=0.3)
ax1.plot(vector[clusters2 == -1], reachability2[clusters2 == -1], 'k.', alpha=0.3)
ax1.plot(vector, np.full_like(vector, 2., dtype=float), 'k-', alpha=0.5)
ax1.plot(vector, np.full_like(vector, 0.5, dtype=float), 'k-.', alpha=0.5)
ax1.set_ylabel('Puntaje de Accecibilidad')
ax1.set_title('Gráfica de Accesibilidad')

# OPTICS Clustering
colors = ['c.', 'b.', 'r.', 'y.', 'g.']
for Class, colour in zip(range(0, 5), colors):
    Xk = X_escalado[modelo_optics2.labels_ == Class]
    ax2.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], colour, alpha=0.3)

ax2.plot(X_escalado.iloc[modelo_optics2.labels_ == -1, 0],
         X_escalado.iloc[modelo_optics2.labels_ == -1, 1],
         'k+', alpha=0.1)
ax2.set_title('OPTICS 2 Clustering')

# DBSCAN Clustering con epsilon = 0.5
colors = ['c', 'b', 'r', 'y', 'g', 'greenyellow']
for Class, colour in zip(range(0, 6), colors):
    Xk = X_escalado[DBSCAN21 == Class]
    ax3.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], colour, alpha=0.3, marker='.')

ax3.plot(X_escalado.iloc[DBSCAN21 == -1, 0],
         X_escalado.iloc[DBSCAN21 == -1, 1],
         'k+', alpha=0.1)
ax3.set_title('DBSCAN Clustering $\epsilon = 0,5$')

# DBSCAN Clustering con epsilon = 2.0
colors = ['c.', 'y.', 'm.', 'g.']
for Class, colour in zip(range(0, 4), colors):
    Xk = X_escalado.iloc[DBSCAN22 == Class]
    ax4.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], colour, alpha=0.3)

ax4.plot(X_escalado.iloc[DBSCAN22 == -1, 0],
         X_escalado.iloc[DBSCAN22 == -1, 1],
         'k+', alpha=0.1)
ax4.set_title('DBSCAN Clustering $\epsilon = 2,0$')

plt.tight_layout()
plt.show()

# Resumen de la data clasificada
final=pd.read_csv('Mall_Customers.csv',sep=';')
final['Cluster'] = clusters1
grouped_data = final.groupby('Cluster')
grouped_data.mean()