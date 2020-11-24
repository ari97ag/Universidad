import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.preprocessing import normalize, StandardScaler

#Carga de la data
X = pd.read_csv('Mall_Customers.csv')

#Limpiando la base de datos
drop_features = ['CustomerID', 'Gender']
print(drop_features)
X = X.drop(drop_features, axis=1)
X.fillna(method='ffill', inplace=True)

#Base limpita
X.head()

# Estandarizando las variables a que sigan un comportameinto normal [0,1]
estandarizado_normal = StandardScaler()
X_estandarizado_normal = estandarizado_normal.fit_transform(X)

# Escalado a [0,1] en las observaciones normalizadas
X_escalado = normalize(X_estandarizado_normal)

X_escalado = pd.DataFrame(X_escalado)
X_escalado.columns = X.columns
X_escalado.head()

# Construcción del modelo de clasificación OPTICS, diferentes metricas
modelo_optic1 = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05,metric='minkowski',p = 1) #Distancia Manhattan
modelo_optic2 = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05,metric='minkowski',p = 2) #Distancia Euclidiana
modelo_optic3 = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05,metric='euclidean')

# Insertando la base en los modelos
modelo_optic1.fit(X_escalado)
modelo_optic2.fit(X_escalado)
modelo_optic3.fit(X_escalado)

# Label según DBSCAN con epsilon = 0.5
labels1 = cluster_optics_dbscan(reachability = modelo_optic1.reachability_, core_distances = modelo_optic1.core_distances_, ordering = modelo_optic1.ordering_, eps = 0.5)
# Label según DBSCAN con epsilon = 2.0
labels2 = cluster_optics_dbscan(reachability = modelo_optic1.reachability_, core_distances = modelo_optic1.core_distances_, ordering = modelo_optic1.ordering_, eps = 2)

# Matriz con posiciones del 0 al 199
space = np.arange(len(X_escalado))

# Distancia de alcance en cada punto observado
reachability = modelo_optic1.reachability_[modelo_optic1.ordering_]
print(reachability)

#Aqui deberia probar todos los criterios com hund y el otro





# Observaciones que logran ser clasificadas con exito y aquellas que no
labels = modelo_optic1.labels_[modelo_optic1.ordering_]
print(labels)

# Definiendo la ventana de visualizacion
plt.figure(figsize=(10, 7))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])

# Plotting the Reachability-Distance Plot
colors = ['c.', 'b.', 'r.', 'y.', 'g.']
for Class, colour in zip(range(0, 5), colors):
    Xk = space[labels == Class]
    Rk = reachability[labels == Class]
    ax1.plot(Xk, Rk, colour, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
ax1.set_ylabel('Distancia de Alcance')
ax1.set_title('Gráfico de Alcance')

# OPTICS Clustering
colors = ['c.', 'b.', 'r.', 'y.', 'g.']
for Class, colour in zip(range(0, 5), colors):
    Xk = X_escalado[modelo_optic1.labels_ == Class]
    ax2.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], colour, alpha=0.3)

ax2.plot(X_escalado.iloc[modelo_optic1.labels_ == -1, 0],
         X_escalado.iloc[modelo_optic1.labels_ == -1, 1],
         'k+', alpha=0.1)
ax2.set_title('OPTICS 1 Clustering')

# DBSCAN Clustering con epsilon = 0.5
colors = ['c', 'b', 'r', 'y', 'g', 'greenyellow']
for Class, colour in zip(range(0, 6), colors):
    Xk = X_escalado[labels1 == Class]
    ax3.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], colour, alpha=0.3, marker='.')

ax3.plot(X_escalado.iloc[labels1 == -1, 0],
         X_escalado.iloc[labels1 == -1, 1],
         'k+', alpha=0.1)
ax3.set_title('DBSCAN Clustering epsilon = 0,5')

# DBSCAN Clustering con epsilon = 2.0
colors = ['c.', 'y.', 'm.', 'g.']
for Class, colour in zip(range(0, 4), colors):
    Xk = X_escalado.iloc[labels2 == Class]
    ax4.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], colour, alpha=0.3)

ax4.plot(X_escalado.iloc[labels2 == -1, 0],
         X_escalado.iloc[labels2 == -1, 1],
         'k+', alpha=0.1)
ax4.set_title('DBSCAN Clustering epsilon = 2,0')

plt.tight_layout()
plt.show()