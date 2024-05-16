
install.packages("bnlearn")
install.packages("gRain")
install.packages("bnviewer")
install.packages("Rmpfr")

install.packages("BiocManager")
BiocManager::install("Rgraphviz")

library(bnlearn)
library(Rgraphviz)
library(bnviewer)
library(gRain)
library(Rmpfr)


## simple network ###
####################

#1) Creating custom fitted Bayesian networks (both the network structure and the parameters are specified by experts/the user)

# create and plot the network structure.
dag1 = model2network("[Water][Running_off_leash][Lepto|Water:Running_off_leash]")
par
graphics::plot.new()
graphviz.plot(dag1)

#specify the conditional probability tables using an expert-driven approach

cpt_ROL = matrix(c(0.4, 0.6), ncol = 2, dimnames = list(NULL, c("TRUE", "FALSE")))
cpt_Water = matrix(c(0.1, 0.9), ncol = 2, dimnames = list(NULL, c("TRUE", "FALSE")))
cpt_Lepto = c(0.8, 0.2, 0.6, 0.4, 0.9, 0.1, 0.5, 0.5)
dim(cpt_Lepto) = c(2, 2, 2)
dimnames(cpt_Lepto) = list("Lepto" = c("TRUE", "FALSE"), "Running_off_leash" =  c("TRUE", "FALSE"),
                      "Water" = c("TRUE", "FALSE"))

cfit = custom.fit(dag1, dist = list(Running_off_leash = cpt_ROL, Water = cpt_Water, Lepto = cpt_Lepto))
graphics::plot.new()
graphviz.chart(cfit, type = "barprob")
graphics::plot.new()
graphviz.chart(cfit, col = "darkblue", bg = "azure", bar.col = "darkblue")



##2) infer parameters from data

#read the data back

#setwd("~/Documents/Lab") ##make sure you change this: use your specific directory where "lepto_data" database is located##
leptodb0<- read.table("lepto_data.txt", header = TRUE, sep = ",")
head(leptodb0)


#make sure you transform the character format
leptodb0[, 'Running_off_leash'] <- as.factor(leptodb0[, 'Running_off_leash'])
leptodb0[, 'Water'] <- as.factor(leptodb0[, 'Water'])
leptodb0[, 'Lepto'] <- as.factor(leptodb0[, 'Lepto'])


#fit the parameters
fitted = bn.fit(dag1, leptodb0)
fitted ##check the probability tables

#plot the graph
graphics::plot.new()
graphviz.chart(fitted, type = "barprob")
graphics::plot.new()
graphviz.chart(fitted, col = "darkblue", bg = "azure", bar.col = "darkblue")




################################
#increase the network complexity
################################

##1) network structure based on experts/user-defined

dag2 = model2network("[Lepto|Water:Running_off_leash][Water|Season:Winter_type][Running_off_leash|Age:Dog_size][Season][Age][Dog_size][Winter_type]")
graphics::plot.new()
graphviz.plot(dag2)


##2) infer parameters from data

#setwd("~/Documents/Lab") ##make sure you change this: use your specific directory##
leptodb<- read.table("lepto_study_final.txt", header = TRUE, sep = ",")
head(leptodb)
summary(leptodb)

#make sure you transform the character format
leptodb[, 'Season'] <- as.factor(leptodb[, 'Season'])
#str(leptodb)  # look at the classes
leptodb[, 'Dog_size'] <- as.factor(leptodb[, 'Dog_size'])
leptodb[, 'Age'] <- as.factor(leptodb[, 'Age'])
leptodb[, 'Winter_type'] <- as.factor(leptodb[, 'Winter_type'])
leptodb[, 'Running_off_leash'] <- as.factor(leptodb[, 'Running_off_leash'])
leptodb[, 'Water'] <- as.factor(leptodb[, 'Water'])
leptodb[, 'Lepto'] <- as.factor(leptodb[, 'Lepto'])



#fit the parameters
fitted = bn.fit(dag2, leptodb)
fitted ##check the conditional probabilities

#plot the graph
graphics::plot.new()
graphviz.chart(fitted, type = "barprob")
graphics::plot.new()
graphviz.chart(fitted, col = "darkblue", bg = "azure", bar.col = "darkblue")




###PERFORM INFERENCES ###

##Inference on Bayesian networks are conducted using conditional probability (CP) or maximum a posteriori (MAP) queries. 

##1) CP query--> we have some evidence (i.e., we know the values of some variables and we fix the nodes accordingly); and
#we want to look into the probability of some event involving the other variables conditional on the evidence we have.

library(gRain)
par(mfrow = c(2, 2))
# 1) Export the BN from bnlearn to gRain, which transforms it into a junction tree.
junction.tree = as.grain(fitted)
# 2) Introduce the evidence in the junction tree, modifying the distribution of Running_off_leash.
with.evidence = setEvidence(junction.tree, nodes = "Running_off_leash", states = "true")
#3) Query about the distribution of the node containing the event of interest.
querygrain(with.evidence, node = "Lepto")

#other example:
with.evidence = setEvidence(junction.tree, nodes = "Dog_size", states = "large")
querygrain(with.evidence, node = "Lepto")


##2) MAP query --> #A MAP query goes further and finds the highest probability in the joint distribution of the variables of interest. 
#goal of a MAP query is to find the combination of values for (a subset of) the variables in the network that has the highest probability given some evidence.

# 2) Introduce the evidence in on Lepto, Season and Running off leash in the junction tree.
with.evidence = setEvidence(junction.tree, nodes = c("Lepto", "Season", "Running_off_leash"),states = c("true", "summer", "true"))
# 3) Query about the distribution of the nodes of interest Age and Winter type
pdist = querygrain(with.evidence, nodes = c("Age","Winter_type"), type = "joint")
# 4) Find the configuration of values with the maximum probability.
which.max = which(pdist == max(pdist), arr.ind = TRUE)
which.max

rownames(pdist)[which.max[, 1]]
colnames(pdist)[which.max[, 2]]



## MEASURING THE STRENGTH OF THE RELATIONSHIP. 

##Measuring the degree of confidence in a particular graphical feature of a Bayesian network is a key problem in the inference on the network structure. In the case of single arcs this quantity is called arc strength.

#p-values from the conditional independence tests that would remove individual arcs present in a network;
pvalues = arc.strength(dag2, data = leptodb, criterion = "x2")

#differences in network score from removing individual arcs present in a network;
score.deltas = arc.strength(dag2, data = leptodb, criterion = "bic")

#probabilities of inclusion of all possible arcs, and their directions;
#Friedman, Goldszmidt and Wyner (1999) introduced a very simple way of quantifying arc strength: generating multiple network structures by applying nonparametric bootstrap to the data and estimating the relative frequency of the feature of interest.
bagging = boot.strength(leptodb, algorithm = "tabu", R = 200)
boot.strength(leptodb, algorithm = "hc", R = 200)
#probabilities of inclusion of individual arcs present in a network, and their directions.
bayes.factors = bf.strength(dag2, data = leptodb)

graphics::plot.new()
par(mfrow = c(2, 2))
graphics::plot.new()
strength.plot(dag2, strength = pvalues, main = "pvalues")
strength.plot(dag2, strength = score.deltas, main = "score.deltas")
strength.plot(dag2, strength = bagging, main = "bagging")
strength.plot(dag2, strength = bayes.factors, main = "bayes.factors")








 
       