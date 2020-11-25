library("opticskxi")
library("dbscan")

### DBSCAN
data('multishapes')
dbscan_shapes <- dbscan::dbscan(multishapes[1:2], eps = 0.15)
gg_shapes <- cbind(multishapes[1:2], Clusters = dbscan_shapes$cluster) %>%
  ggpairs(group = 'Clusters')

data('DS3', package = 'dbscan')
dbscan_ds3 <- dbscan::dbscan(DS3, minPts = 25, eps = 12) 
gg_ds3 <- cbind(DS3, Clusters = dbscan_ds3$cluster) %>%
  ggpairs(group = 'Clusters') 
cowplot::plot_grid(gg_shapes, gg_ds3, nrow = 2, labels = c('(a)', '(b)'), label_x = 0.9)

n<-1e3
set.seed(0)
multi_gauss <- cbind.data.frame(
  x = c(rnorm(n / 2, -3), rnorm(n / 4, 3), rnorm(n / 4, 3, .2)),
  y = c(rnorm(n * .75), rnorm(n / 8, 1, .2), rnorm(n / 8, -1, .2)))

dbscan_gauss <- dbscan::dbscan(multi_gauss, minPts = 30, eps = .5)
gg_mgauss <- cbind(multi_gauss, Clusters = dbscan_gauss$cluster) %>%
  ggpairs(group = 'Clusters')

gg_mgauss_small <- dbscan::dbscan(multi_gauss, minPts = 30, eps = .2) %$%
  cbind(multi_gauss, Clusters = cluster) %>% ggpairs(group = 'Clusters') 
cowplot::plot_grid(gg_mgauss, gg_mgauss_small, nrow = 2, labels = c('(a)', '(b)'), label_x = .9)

### OPTICS
optics_gauss <- dbscan::optics(multi_gauss, minPts = 30)
xi_gauss <- dbscan::extractXi(optics_gauss, xi = 0.03)
ggplot_optics(optics_gauss, groups = xi_gauss$cluster)

kxi_gauss <- opticskxi(optics_gauss, n_xi = 4, pts = 100)
ggplot_optics(optics_gauss, groups = kxi_gauss)

gg_shapes_optics <- dbscan::optics(multishapes[1:2]) %>%
  ggplot_optics(groups = opticskxi(., n_xi = 5, pts = 30))

gg_ds3_optics <- dbscan::optics(DS3, minPts = 25) %>%
  ggplot_optics(groups = opticskxi(., n_xi = 6, pts = 100))
cowplot::plot_grid(gg_shapes_optics, gg_ds3_optics, nrow = 2, labels = c('(a)', '(b)'), label_x = .9)

library("fpc")
library("gap")
library("amap")
library("fastICA")
library("reshape2")

setwd("/Users/ari97ag/Documents/GitHub/Universidad/OPTICS/R/")
hla<-read.csv("Mall_Customers.csv", header=TRUE,sep=",", na.strings="NA")
attach(hla)
#data('hla')
m_hla <- hla[-c(1:2)] %>% scale
df_params_hla <- expand.grid(n_xi = 3:5, pts = c(10, 10, 10), dist = c('manhattan', 'euclidean', 'abscorrelation', 'abspearson'))
df_kxi_hla <- opticskxi_pipeline(m_hla, df_params_hla)
ggplot_kxi_metrics(df_kxi_hla, n = 8)

gtable_kxi_profiles(df_kxi_hla) %>% plot

best_kxi_hla <- get_best_kxi(df_kxi_hla, rank = 2)
clusters_hla <- best_kxi_hla$clusters

kk<-rep(1,200)
hla2<-cbind(hla,clusters_hla,kk)
aggregate(Gender, by=list(hla2$clusters_hla), sum)

fortify_pca(m_hla, sup_vars = data.frame(Clusters = clusters_hla)) %>% ggpairs('Clusters', ellipses = TRUE, variables = TRUE)
