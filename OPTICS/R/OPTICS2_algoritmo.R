library("opticskxi")
library("dbscan")

setwd("/Users/ari97ag/Documents/GitHub/Universidad/OPTICS/R/")
base<-read.csv("Mall_Customers.csv", header=TRUE,sep=",", na.strings="NA")
attach(base)
base_1 <- base[-c(1:2)] %>% scale

#DBSCAN

modelDBSCAN1<-dbscan(base_1,eps=0.6, MinPts = 8,borderPoints = TRUE)
head(cbind(base_1,modeloDBSCAN1$cluster))
hullplot(base_1,modeloDBSCAN1$cluster, main = "Convex cluster Hulls, eps= 0.4")

#OPTICS
opticskxi_pipeline(base_1, 
                   df_params = expand.grid(n_xi = 1:10, pts =c(20, 30, 40), 
                                           dist = c("euclidean", "abscorrelation"), 
                                           dim_red =c("identity", "PCA", "ICA"), 
                                           n_dimred_comp = c(5, 10, 20)),
                   n_cores = 1)



modelOPTICS<-optics(base_1, eps=10, minPts = 10)

kk<-extractXi(modelOPTICS,xi=0.05)
kk$clusters_xi
hullplot(base_1,kk)

dend <- as.dendrogram(kk);dend
plot(dend, ylab = "Reachability dist.", leaflab = "none")


library("fpc")
library("gap")
library("amap")
library("fastICA")
library("reshape2")
qq<-data('hla')
m_hla <- hla[-c(1:2)] %>% scale
df_params_hla <- expand.grid(n_xi = 3:5, pts = c(20, 30),
                             dist = c('manhattan', 'euclidean'))
df_kxi_hla <- opticskxi_pipeline(m_hla, df_params_hla)
ggplot_kxi_metrics(df_kxi_hla, n = 8)
gtable_kxi_profiles(df_kxi_hla) %>% plot
best_kxi_hla <- get_best_kxi(df_kxi_hla, rank = 2)
clusters_hla <- best_kxi_hla$clusters

fortify_pca(m_hla, sup_vars = data.frame(Clusters = clusters_hla)) %>%
  ggpairs('Clusters', ellipses = TRUE, variables = TRUE)

#Pipeline me da la distancia de las metricas

#Para agregar y categorizar la base
kk<-rep(1,200)
hla2<-cbind(hla,clusters_hla,kk)
aggregate(kk, by=list(hla2$clusters_hla), sum)

