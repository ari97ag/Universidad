library("opticskxi")
library("dbscan")
library("dplyr")
library("clusterSim")
library("fossil")
library("MixGHD")
library("fpc")

setwd("/Users/ari97ag/Documents/GitHub/Universidad/OPTICS/R/")
base<-read.csv("Mall_Customers.csv", header=TRUE,sep=";", na.strings="NA")
attach(base)
base_1 <- data.frame(base[-c(1:2)] %>% scale)

#minPts optimo
#k: numero de dimension de la base + 1 seria el minPts.
kNNdistplot(base_1, k = 4)
abline(h=.55, col = "red", lty=2)

# DBSCAN
dbscan_1 <- dbscan(base_1, MinPts = 4, eps = 0.55)
gg_dbscan_1 <- cbind(base_1, Clusters = dbscan_1$cluster) %>%
  ggpairs( group = "Clusters", ellipses = T)
gg_dbscan_1

# OPTICS
# n_xi: Cantidad de clusters que espero
# pts: Observaciones minimas dentro del nucleo del cluster
# dist: Distancias

#Distancia Euclidiana
optics_parametros_1<-expand.grid(n_xi = 3:6,pts = c(4,6,8,10),dist = "euclidean")
modelos_optics1 <- opticskxi_pipeline(base_1, optics_parametros_1)
ggplot_kxi_metrics(modelos_optics1,n = 12)

#Grafica de accesibilidad de los 4 mejores de los modelos
gtable_kxi_profiles(modelos_optics1, rank = 1:4) %>% plot

#Distancia Manhattan
optics_parametros_2<-expand.grid(n_xi = 3:6, pts = c(4,6,8,10),dist = "manhattan")
modelos_optics2 <-opticskxi_pipeline(base_1, optics_parametros_2)
ggplot_kxi_metrics(modelos_optics2, n = 12)

#Grafica de accesibilidad de los 4 mejores modelos
gtable_kxi_profiles(modelos_optics1, rank = 1:4) %>% plot
gtable_kxi_profiles(modelos_optics2, rank = 1:4) %>% plot

#Seleccion del mejor modelo optics
mejor_modelo_optics1 <- get_best_kxi(modelos_optics1, rank = 1);mejor_modelo_optics1 
mejor_modelo_optics2 <- get_best_kxi(modelos_optics1, rank = 2);mejor_modelo_optics2
mejor_modelo_optics3 <- get_best_kxi(modelos_optics2, rank = 1);mejor_modelo_optics3
mejor_modelo_optics4 <- get_best_kxi(modelos_optics2, rank = 2);mejor_modelo_optics4

clusters1 <- mejor_modelo_optics1$clusters;clusters1
clusters2 <- mejor_modelo_optics2$clusters;clusters2
clusters3 <- mejor_modelo_optics3$clusters;clusters3
clusters4 <- mejor_modelo_optics4$clusters;clusters4

clusters1[1]=0 #NA
clusters2[1]=0 #NA
clusters3[1]=0 #NA
clusters4[1]=0 #NA

# Coeficiente de Davies-Bouldin
print(index.DB(base_1, clusters1, centrotypes="centroids",p=2))
print(index.DB(base_1, clusters2, centrotypes="centroids",p=2))
print(index.DB(base_1, clusters3, centrotypes="centroids",p=1))
print(index.DB(base_1, clusters4, centrotypes="centroids",p=1))

# Clustering OPTICS 4
base_2<-cbind(base,clusters4)
kk<-base_2[order(base_2$clusters4),]
base_3 <- base_2[-c(1:2)]
grupos<-aggregate(. ~ clusters4, base_3, mean);grupos

fortify_pca(base_1, sup_vars = data.frame(Clusters = clusters4)) %>% 
  ggpairs('Clusters', ellipses = T)

# VALIDACION EXTERNA
x_referencia<-dbscan_1$cluster
x_comparativo<-clusters4

# Indice de Rand
rand.index(x_referencia, x_comparativo)
# Indice de Rand Ajustado
ARI(x=x_referencia, y=x_comparativo)

# Más metricas comparativas
clust_stats <- cluster.stats(d = dist(base_1), x_referencia, x_comparativo)
clust_stats$corrected.rand
clust_stats$vi
