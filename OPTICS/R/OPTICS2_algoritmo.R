library("fpc")
library("gap")
library("amap")
library("fastICA")
library("reshape2")
library("opticskxi")

setwd("/Users/ari97ag/Documents/GitHub/Universidad/OPTICS/R/")
base<-read.csv("Mall_Customers.csv", header=TRUE,sep=",", na.strings="NA")
attach(base)
base_1 <- base[-c(1:2)] %>% scale

# n_xi: Cantidad de clusters que espero
# pts: Observaciones minimas dentro del nucleo del cluster
# dist: Distancias
optics_parametros_1<-expand.grid(n_xi = 3:6, pts = c(20,30),dist = "euclidean")
optics_parametros_2<-expand.grid(n_xi = 3:6, pts = c(20,30),dist = "manhattan")

modelos_optics1 <- opticskxi_pipeline(base_1, optics_parametros_1, n_cores = 1)
ggplot_kxi_metrics(modelos_optics1, n = 8)

modelos_optics2 <-opticskxi_pipeline(base_1, optics_parametros_2, n_cores = 1)
ggplot_kxi_metrics(modelos_optics2, n = 8)

#Grafica de alcanzabilidad de los 4 mejores de los modelos con distancia de manhattan
gtable_kxi_profiles(modelos_optics1) %>% plot

#Seleccion del mejor modelo optics
mejor_modelo_optics <- get_best_kxi(modelos_optics1, rank = 1)
clusters <- mejor_modelo_optics$clusters;clusters

#Clustering por sexo
qq<-rep(1,200)
base_2<-cbind(base_1,clusters,qq)
aggregate(qq, by=list(clusters,Gender), sum)

fortify_pca(base_1, sup_vars = data.frame(Clusters = clusters)) %>% 
  ggpairs('Clusters', ellipses = TRUE)

x_referencia
x_comparativo

# Para realizar una validacion externa podriamos hacerlo si tenemos los 
# datos categorizados originales de la base o por medio de la comparacion con otro metodo o modelo
library("fossil")
#indice de Rand
rand.index(x_referencia, x_comparativo)

