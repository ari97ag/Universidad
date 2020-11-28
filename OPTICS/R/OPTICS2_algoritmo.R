library("opticskxi")
library("dplyr")

setwd("/Users/ari97ag/Documents/GitHub/Universidad/OPTICS/R/")
base<-read.csv("Mall_Customers.csv", header=TRUE,sep=";", na.strings="NA")
attach(base)
base_1 <- base[-c(1:2)] %>% scale

# n_xi: Cantidad de clusters que espero
# pts: Observaciones minimas dentro del nucleo del cluster
# dist: Distancias

#Distancia Euclidiana
optics_parametros_1<-expand.grid(n_xi = 3:5, pts = c(10,30),dist = "euclidean")
modelos_optics1 <- opticskxi_pipeline(base_1, optics_parametros_1)
ggplot_kxi_metrics(modelos_optics1,n = 6)

#Grafica de alcanzabilidad de los 4 mejores de los modelos
gtable_kxi_profiles(modelos_optics1, rank = 1:4) %>% plot

#Distancia Manhattan
optics_parametros_2<-expand.grid(n_xi = 3:5, pts = c(10,30),dist = "manhattan")
modelos_optics2 <-opticskxi_pipeline(base_1, optics_parametros_2)
ggplot_kxi_metrics(modelos_optics2, n = 6)

#Grafica de alcanzabilidad de los 4 mejores de los modelos
gtable_kxi_profiles(modelos_optics2, rank = 1:4) %>% plot

#Seleccion del mejor modelo optics
mejor_modelo_optics1 <- get_best_kxi(modelos_optics1, rank = 1)
mejor_modelo_optics2 <- get_best_kxi(modelos_optics2, rank = 1)
clusters1 <- mejor_modelo_optics1$clusters;clusters
clusters2 <- mejor_modelo_optics2$clusters;clusters

#Clustering OPTICS 1
base_2<-cbind(base,clusters)
kk<-base_2[order(base_2$clusters),]
base_3 <- base_2[-c(1:2)]
grupos<-aggregate(. ~ clusters, base_3, mean);grupos

fortify_pca(base_1, sup_vars = data.frame(Clusters = clusters1)) %>% 
  ggpairs('Clusters', ellipses = T)

#VALIDACION EXTERNA
x_referencia
x_comparativo

# Para realizar una validacion externa podriamos hacerlo si tenemos los 
# datos categorizados originales de la base o por medio de la comparacion con otro metodo o modelo
library("fossil")
#indice de Rand
rand.index(x_referencia, x_comparativo)