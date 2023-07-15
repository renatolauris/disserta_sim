rm(list = ls())

# loading libraries -------------------------------------------------------

library(grf)
# library(mgcv)
# library(randomForestCI) #computing causalForest variance
library(FNN)
library(Hmisc)
library(xtable)
library(pacman)
p_load(tidyverse)
p_load(np)
p_load(writexl)
p_load(ranger)
library(hdm)
library(glmnet)
library(purrr)
require(splines)
library(readxl)

library(DoubleML)
library(mlr3)
library(mlr3learners)
library(data.table)
library(ggplot2)
library(reticulate)
# citation("FNN")
# citation("grf")
# citation("np")

# 0. loading working directory and data ----------------------------------------------------------

getwd()
end.temp <- "C:/Users/renat/Dropbox/Dissertacao-Renato/2.simulacoes de causalidade"
setwd(end.temp)

# 2a. KNN estimator ---------------------------------------

# knn causal function
causal.kn <- function(kn,X,W,X.test,Y){
  # function inputs
  # kn: K-nearest neighbour
  # X: vector of covariates, train data
  # W: treatment variable, train data
  # Y: outcome variable, train data
  # X.test: vector of covariates, test data
  
  knn.0.mu = knn.reg(X[W==0,], X.test, Y[W==0], k = kn)$pred
  knn.1.mu = knn.reg(X[W==1,], X.test, Y[W==1], k = kn)$pred
  
  knn.0.mu2 = knn.reg(X[W==0,], X.test, Y[W==0]^2, k = kn)$pred
  knn.1.mu2 = knn.reg(X[W==1,], X.test, Y[W==1]^2, k = kn)$pred
  
  knn.0.var = (knn.0.mu2 - knn.0.mu^2) / (kn - 1)
  knn.1.var = (knn.1.mu2 - knn.1.mu^2) / (kn - 1)
  
  knn.tau = knn.1.mu - knn.0.mu
  knn.se = sqrt(knn.0.var + knn.1.var)
  cbind(knn.tau,knn.se) %>% as.data.frame()
  # function outputs:
  # knn.tau: K-nearest neighbour tau estimate
  # knn.se: K-nearest neighbour standard errors of tau estimate
}

# 2b. Inverse propensity-weighted estimator ------------------------------

# adjust X vector to model matrix
make_matrix = function(x) stats::model.matrix(~.-1, x)
# transform X vector to splines for lasso regression
make_matrix_splines = function(x) {
  col=1
  X_ns = do.call(cbind, lapply(1:dim(x)[2], function(col){matrix(splines::ns(x[,col],df=7), nrow(x), 7)}))
  dim_ns = dim(X_ns)[2]
  X_ns = stats::model.matrix(~.*.-1, data.frame(X_ns)) # pairwise interaction (not including squared term for each column)
  X_ns_sq = do.call(cbind, lapply(1:dim_ns, function(col){matrix(X_ns[,col]^2)})) # squared term for each column
  X_ns = cbind(X_ns, X_ns_sq)
  make_matrix(data.frame(X_ns))
}

# function to trim the propensity score function probability space
# to hold common support hypotesis
trimmed_ps <- function(x,p=0.01){
  x = ifelse(x<p, p, ifelse(x>1-p,1-p, x))
}

# ipw estimator by random forest regression with predetermined ps threshold
ipw.rf.estimator <- function(X,Y,W,n_fold=5,p_threshold=0.01){
  # function inputs
  # X: vector of covariates, train data
  # W: treatment variable, train data
  # Y: outcome variable, train data
  # n_fold: n fold for cross-fit first stage estimations
  # p_threshold: threshold for trimmed propensity score
  
  # first stage
  # A list of vectors indicating the left-out subset
  n <- nrow(X)
  n.folds <- n_fold
  # indices <- split(seq(n), sort(seq(n) %% n.folds))
  foldid <- rep.int(1:n.folds,times = ceiling(n/n.folds))[sample.int(n)] #define folds indices
  indices <- split(1:n, foldid)  #split observation indices into folds  
  ipw.scores <- lapply(indices, function(idx) {
    
    # Fitting the propensity score model
    # Comment / uncomment the lines below as appropriate.
    # OBSERVATIONAL SETTING (with unconfoundedness+overlap):
    model.e <- ranger(W~., max.depth=8, data=data.frame(cbind(W=W[-idx],X[-idx,])))  
    e.hat <- predict(model.e, X[idx,], type="response")$predictions
    e.hat <- trimmed_ps(e.hat,p=p_threshold)
    # RANDOMIZED SETTING
    # e.hat <- rep(0.5, length(idx))  
    
    
    # Compute IPW scores
    ipw.scores <- Y[idx] * (W[idx]/e.hat - (1-W[idx])/(1-e.hat))
    ipw.scores
  })
  ipw.scores<- unname(do.call(c, ipw.scores))
  
  # second stage
  tau.scores <- lapply(indices, function(idx) {
    # Fitting the outcome model
    model.e <- ranger(W[-idx]~., max.depth=8, data=data.frame(cbind(W[-idx],X[-idx,])))  
    
    outcome.model <- ranger(Y~., max.depth=8, data=data.frame(cbind(Y=ipw.scores[-idx],X[-idx,])))
    tau.hat <- predict(outcome.model, X[idx,], type="response")$predictions
    tau.hat
  })
  tau.scores<- unname(do.call(c, tau.scores))
  return(tau.scores)
  # function outputs:
  # tau.scores: IPW with randomforest tau estimate
}

# ipw estimator by lasso regression with predetermined ps threshold
ipw.lasso.estimator <- function(X,Y,W,n_fold=5,p_threshold=0.01){
  # function inputs
  # X: vector of covariates, train data
  # W: treatment variable, train data
  # Y: outcome variable, train data
  # n_fold: n fold for cross-fit first stage estimations
  # p_threshold: threshold for trimmed propensity score
  
  # first stage
  XX <- make_matrix_splines(X)
  # A list of vectors indicating the left-out subset
  n <- nrow(XX)
  n.folds <- n_fold
  # indices <- split(seq(n), sort(seq(n) %% n.folds))
  foldid <- rep.int(1:n.folds,times = ceiling(n/n.folds))[sample.int(n)] #define folds indices
  indices <- split(1:n, foldid)  #split observation indices into folds  
  ipw.scores <- lapply(indices, function(idx) {
    
    # Fitting the propensity score model
    # Comment / uncomment the lines below as appropriate.
    # OBSERVATIONAL SETTING (with unconfoundedness+overlap):
    model.e <- cv.glmnet(XX[-idx,], W[-idx], family="binomial")  
    e.hat <- predict(model.e, XX[idx,], s="lambda.min", type="response")
    e.hat <- trimmed_ps(e.hat,p=p_threshold)
    # RANDOMIZED SETTING
    # e.hat <- rep(0.5, length(idx))  
    
    # Compute IPW scores
    ipw.scores <- Y[idx] * (W[idx]/e.hat - (1-W[idx])/(1-e.hat))
    
    ipw.scores
  })
  ipw.scores<- unname(do.call(c, ipw.scores))
  
  # second stage
   tau.scores <- lapply(indices, function(idx) {
    # Fitting the outcome model
    outcome.model <- cv.glmnet(x=XX[-idx,], y=ipw.scores[-idx], family="gaussian")
    tau.hat <- predict(outcome.model, XX[idx,], s = "lambda.min", type="response")
    tau.hat
  })
  tau.scores<- unname(do.call(c, tau.scores))
  return(tau.scores)
  # function outputs:
  # tau.scores: IPW with lasso regression tau estimate
  
}

# 2c. Augmented inverse propensity-weighted (AIPW) estimator -------------

# aipw estimator by random forest regression with predetermined ps threshold
aipw.rf.estimator <- function(X,Y,W,n_fold=5,p_threshold=0.01){
  # function inputs
  # X: vector of covariates, train data
  # W: treatment variable, train data
  # Y: outcome variable, train data
  # n_fold: n fold for cross-fit first stage estimations
  # p_threshold: threshold for trimmed propensity score
  
  # first stage
  
  # A list of vectors indicating the left-out subset
  n <- nrow(X)
  n.folds <- n_fold
  # indices <- split(seq(n), sort(seq(n) %% n.folds))
  foldid <- rep.int(1:n.folds,times = ceiling(n/n.folds))[sample.int(n)] #define folds indices
  indices <- split(1:n, foldid)  #split observation indices into folds  
  
  # Preparing data
  Y <- Y
  W <- W
  X <- X
  data <- data.frame(Y,W,X)
  covariates <- paste0("X",1:ncol(X))
  treatment <- "W"
  
  # # Matrix of (transformed) covariates used to estimate E[Y|X,W]
  # fmla.xw <- formula(paste("~ 0 +", paste0("bs(", covariates, ", df=3)", "*", treatment, collapse=" + ")))
  # XW <- model.matrix(fmla.xw, data)
  XW <- data.frame(cbind(X,W))
  # Matrix of (transformed) covariates used to predict E[Y|X,W=w] for each w in {0, 1}
  data.1 <- XW
  data.1[,treatment] <- 1
  XW1 <- data.1
  # XW1 <- model.matrix(fmla.xw, data.1)  # setting W=1
  data.0 <- XW
  data.0[,treatment] <- 0
  XW0 <- data.0
  # XW0 <- model.matrix(fmla.xw, data.0)  # setting W=0
  
  # # Matrix of (transformed) covariates used to estimate and predict e(X) = P[W=1|X]
  # fmla.x <- formula(paste(" ~ 0 + ", paste0("bs(", covariates, ", df=3)", collapse=" + ")))
  # XX <- model.matrix(fmla.x, data)
  # 
  # Cross-fitted estimates of E[Y|X,W=1], E[Y|X,W=0] and e(X) = P[W=1|X]
  mu.hat.1 <- rep(NA, n)
  mu.hat.0 <- rep(NA, n)
  e.hat <- rep(NA, n)
  
  for (idx in indices) {
    # Estimate outcome model and propensity models
    # Note how cross-validation is done (via cv.glmnet) within cross-fitting! 
    outcome.model <- ranger(Y~., max.depth=8, data=data.frame(cbind(Y=Y[-idx],XW[-idx,])))
    propensity.model <- ranger(W~., max.depth=8, data=data.frame(cbind(W=W[-idx],X[-idx,])))

    # Predict with cross-fitting
    mu.hat.1[idx] <- predict(outcome.model, XW1[idx,], type="response")$predictions
    mu.hat.0[idx] <- predict(outcome.model, XW0[idx,], type="response")$predictions
    e.hat[idx] <- predict(propensity.model, X[idx,], type="response")$predictions
  }
  
  # Commpute the summand in AIPW estimator
  aipw.scores <- (mu.hat.1 - mu.hat.0
                  + W / e.hat * (Y -  mu.hat.1)
                  - (1 - W) / (1 - e.hat) * (Y -  mu.hat.0))
  
  # second stage
  tau.scores <- lapply(indices, function(idx) {
    # Fitting the outcome model
    outcome.model <- ranger(Y~., max.depth=8, data=data.frame(cbind(Y=aipw.scores[-idx],XW[-idx,])))
    tau.hat <- predict(outcome.model, XW[idx,],type="response")$predictions
    tau.hat
  })
  tau.scores<- unname(do.call(c, tau.scores))
  return(tau.scores)
  # function outputs:
  # tau.scores: IPW with lasso regression tau estimate
}

# aipw estimator by lasso regression with predetermined ps threshold
aipw.lasso.estimator <- function(X,Y,W,n_fold=5,p_threshold=0.01){
  # function inputs
  # X: vector of covariates, train data
  # W: treatment variable, train data
  # Y: outcome variable, train data
  # n_fold: n fold for cross-fit first stage estimations
  # p_threshold: threshold for trimmed propensity score
  
  # first stage
  
  XX <- make_matrix_splines(X)
  # A list of vectors indicating the left-out subset
  n <- nrow(XX)
  n.folds <- n_fold
  # indices <- split(seq(n), sort(seq(n) %% n.folds))
  foldid <- rep.int(1:n.folds,times = ceiling(n/n.folds))[sample.int(n)] #define folds indices
  indices <- split(1:n, foldid)  #split observation indices into folds  
  
  # Preparing data
  Y <- Y
  W <- W
  X <- X
  data <- data.frame(Y,W,X)
  covariates <- paste0("X",1:ncol(X))
  treatment <- "W"
  
  # Matrix of (transformed) covariates used to estimate E[Y|X,W]
  fmla.xw <- formula(paste("~ 0 +", paste0("bs(", covariates, ", df=3)", "*", treatment, collapse=" + ")))
  XW <- model.matrix(fmla.xw, data)
  # Matrix of (transformed) covariates used to predict E[Y|X,W=w] for each w in {0, 1}
  data.1 <- data
  data.1[,treatment] <- 1
  XW1 <- model.matrix(fmla.xw, data.1)  # setting W=1
  data.0 <- data
  data.0[,treatment] <- 0
  XW0 <- model.matrix(fmla.xw, data.0)  # setting W=0
  
  # Matrix of (transformed) covariates used to estimate and predict e(X) = P[W=1|X]
  fmla.x <- formula(paste(" ~ 0 + ", paste0("bs(", covariates, ", df=3)", collapse=" + ")))
  XX <- model.matrix(fmla.x, data)
  
  # (Optional) Not penalizing the main effect (the coefficient on W)
  penalty.factor <- rep(1, ncol(XW))
  penalty.factor[colnames(XW) == treatment] <- 0
  
  # Cross-fitted estimates of E[Y|X,W=1], E[Y|X,W=0] and e(X) = P[W=1|X]
  mu.hat.1 <- rep(NA, n)
  mu.hat.0 <- rep(NA, n)
  e.hat <- rep(NA, n)
  for (idx in indices) {
    # Estimate outcome model and propensity models
    # Note how cross-validation is done (via cv.glmnet) within cross-fitting! 
    outcome.model <- cv.glmnet(x=XW[-idx,], y=Y[-idx], family="gaussian", penalty.factor=penalty.factor)
    propensity.model <- cv.glmnet(x=XX[-idx,], y=W[-idx], family="binomial")
    
    # Predict with cross-fitting
    mu.hat.1[idx] <- predict(outcome.model, newx=XW1[idx,], type="response")
    mu.hat.0[idx] <- predict(outcome.model, newx=XW0[idx,], type="response")
    e.hat[idx] <- predict(propensity.model, newx=XX[idx,], type="response")
  }
  
  # Commpute the summand in AIPW estimator
  aipw.scores <- (mu.hat.1 - mu.hat.0
                  + W / e.hat * (Y -  mu.hat.1)
                  - (1 - W) / (1 - e.hat) * (Y -  mu.hat.0))
  
  # second stage
  tau.scores <- lapply(indices, function(idx) {
    # Fitting the outcome model
    outcome.model <- cv.glmnet(x=XX[-idx,], y=aipw.scores[-idx], family="gaussian")
    tau.hat <- predict(outcome.model, XX[idx,], s = "lambda.min", type="response")
    tau.hat
  })
  tau.scores<- unname(do.call(c, tau.scores))
  return(tau.scores)
  # function outputs:
  # tau.scores: IPW with lasso regression tau estimate
}

# 2d. DML estimator ------------------------------------------------------

# estimate W residual
wreg <- function(X_train,W,X_test){
  wfit=ranger(W~., max.depth=8, data=data.frame(cbind(W,X_train))) 
  predict(wfit, data=X_test)$predictions
} #ML method=Forest 

# estimate Y residual
yreg <- function(X_train,Y,X_test){
  yfit=ranger(Y~., max.depth=8, data=data.frame(cbind(Y,X_train))) 
  predict(yfit, data=X_test)$predictions
} #ML method=Forest 

# cross-fitting residuals
DML.LearnResiduals <- function(X, W, Y, nfold=5) {
  nobs <- nrow(X) #number of observations
  foldid <- rep.int(1:nfold,times = ceiling(nobs/nfold))[sample.int(nobs)] #define folds indices
  Id <- split(1:nobs, foldid)  #split observation indices into folds  
  ytil <- wtil <- rep(NA, nobs)
  # cat("fold: ")
  for(b in 1:length(Id)){
    what <- wreg(X_train=X[-Id[[b]],], W=W[-Id[[b]]], X_test = X[Id[[b]],] ) #take a fold out
    yhat <- yreg(X_train=X[-Id[[b]],], Y=Y[-Id[[b]]], X_test = X[Id[[b]],] ) # take a fold out
    wtil[Id[[b]]] <- (W[Id[[b]]] - what) #record residual for the left-out fold
    ytil[Id[[b]]] <- (Y[Id[[b]]] - yhat) #record residual for the left-out fold
    # cat(b," ")
  }
  # 
  # cat(sprintf("Controls explain %g per cent of variance of Outcome",  round( max(1-var(ytil)/var(y),0)*100, digits=3)) )
  # cat(sprintf("\n Controls explain %g per cent of variance of Treatment",  round( max(1-var(wtil)/var(W),0)*100, digits=3)) )
  return( list(wtil=wtil, ytil=ytil) ) #save output and residuals 
}

# double lasso regression ytil on wtil 
# with b term as covariates as main outcome and multiplied on wtil 
DML.CATE.DLasso<- function(wtil, ytil, b, name="wtil"){
  # ytil is y resid; #wtil is W resid
  # name is name of variable whose coefficient we want to infer
  # name could be a list of names
  # name=grep("wtil", colnames(X.lasso))[1:3] will give the first three coefficiens 
  ytil = ytil
  wtil = wtil
  b <- b[, which(apply(b, 2, var) != 0)] # exclude all constant variables
  demean<- function (x){ x- mean(x)}
  b<- apply(as.matrix(b), 2, FUN=demean)
  X.lasso= model.matrix( ~ wtil+ wtil:b + b)
  # index.treatment = name 
  # if all interactons are of interest write: 
  index.treatment = grep(name, colnames(X.lasso))
  effects.treatment <- rlassoEffects(x = X.lasso, y = ytil, index = index.treatment, post=FALSE)
  result=summary(effects.treatment)
  return(coef=result$coef)
}

# double/debiased machine learning estimator
dml.estimator <- function(Y,W,X){
  # function inputs
  # X: vector of covariates, test data
  # W: treatment variable, test data
  # Y: outcome variable, test data
    set.seed(1)
  res <- DML.LearnResiduals(X,W,Y)
  B=X
  result <- DML.CATE.DLasso(wtil=res$wtil,ytil=res$ytil, b=B,
                            name="wtil")
  B <- B[, which(apply(B, 2, var) != 0)] # exclude all constant variables
  data <- cbind(Y=Y,W=W,B) %>% as.data.frame()
  data2 <- stats::model.matrix(~1 + ., data.frame(data[,-c(1,2)]))
  tau_hat <- data2 %*% result[,1]%>% as.vector()
  se_hat <- data2 %*% result[,2]%>% as.vector()
  return( list(tau_hat=tau_hat, se_hat=se_hat) )
  # function outputs:
  # tau_hat: dml tau estimate
  # se_hat: dml standard errors of tau estimate
}

# import function created in python for estimate CATE by DoubleML package
# Python: Conditional Average Treatment Effects (CATEs) — DoubleML documentation
# https://docs.doubleml.org/stable/examples/py_double_ml_cate.html
# Sys.which("python")
# use_python("C:\\Users\\renat\\AppData\\Local\\Programs\\Python\\PYTHON~1\\python.exe")
source_python("dml.py")
# dml_py(data_train_full = dados_train_full,
#                 data_train = dados_train,
#                 outcome = "Y",treatment = "W",
#                 covariates = cova_train,data_test = dados_test)
# function inputs
# data_train_full: vector of outcome + treatment + covariates (this order), train data
# data_train: vector of covariates, X_train, train data
# outcome: outcome variable name
# treatment: treatment variable name
# covariates: vector of covariates name
# data_test: vector of covariates, X_test, test data

# function outputs:
# tau_hat: dml tau estimate
# se_hat: dml standard errors of tau estimate


# SIMULATION --------------------------------------------------------------

simu.fun = function(n,d,sigma,setup) {
  # function inputs
  # n: sample size
  # d: number of covariates
  # sigma: variance of error term
  # setup: data generating process (DGP) setup choice
  
  # function to choose the data generating process (DGP) parameters
  # the output is a list with X=X, b=b, tau=tau, e=e parameters
  if (setup == 'A') {#linear setup
    get.params = function() {
      X = matrix(runif(n * d, min=0, max=1), n, d) # covariates
      b = X[,1] + 0.5*X[,2] + 0.4*X[,3] # baseline main effect
      tau = 0.3*X[,1] + 0.6*X[,2] + 0.6*X[,3] + 2.2*X[,4] # effect = tau(CATE)
      eta = 0.1
      e = pmax(eta, pmin(sin(pi * X[,1] * X[,2]), 1-eta)) #propensity score trimmed eta 0.1
      list(X=X, b=b, tau=tau, e=e)
    }
  } else if (setup == 'B') {# nonlinear setup
    get.params = function() {
      X = matrix(runif(n*d, min=0, max=1), n, d) # covariates
      b = 0.2*X[,1] + X[,1]^2 + 0.5*X[,2]*X[,3] + 0.4*X[,3] + 0.8*X[,3]^2 # baseline main effect
      eta = 0.1
      e = pmax(eta, pmin(X[,1]- 0.2*X[,2]^2, 1-eta)) #propensity score trimmed eta 0.1
      tau = 0.3*X[,1] + 0.6*X[,2] + 0.6*X[,3] + 2.2*X[,4]  # effect = tau(CATE)
      list(X=X, b=b, tau=tau, e=e)
    }
  } else if (setup == 'C') {# peaks and valleys setup
    get.params = function() {
      X = matrix(runif(n * d, min=0, max=1), n, d) # covariates
      b = 0 # baseline main effect
      tau = (1 + 1 / (1 + exp(-20 * (X[,1] - 1/3)))) * (1 + 1 / (1 + exp(-20 * (X[,2] - 1/3))))* (1 + 1 / (1 + exp(-20 * (X[,3] - 1/3)))) # effect = tau(CATE)
      e = 0.5*X[,1] + 0.5*X[,2] #propensity score
      list(X=X, b=b, tau=tau, e=e)
    }
  } else if (setup == 'D') {# discontinuities setup
    get.params = function() {
      X = matrix(rnorm(n * d), n, d) # covariates
      b = 2 * (X[,1]>0.4) + 0.3*X[,2] # baseline main effect
      eta = 0.1
      e = pmax(eta, pmin(0.7*X[,1]- 0.7*X[,2]+ 0.7*X[,3], 1-eta)) #propensity score trimmed eta 0.1
      tau = 2 * (X[,1]>0.6) + 1.5 * (X[,2]>0.6) + 0.3*X[,3] - 0.7*X[,4]  # effect = tau(CATE)
      list(X=X, b=b, tau=tau, e=e)
    }
  } else {
    
    stop("bad setup")
  }
  params_train = get.params()
  W_train = rbinom(n, 1, params_train$e)
  Y_train = params_train$b + (W_train - 0.5) * params_train$tau + sigma * rnorm(n)
  
  params_test = get.params()
  W_test = rbinom(n, 1, params_test$e)
  Y_test = params_test$b + (W_test - 0.5) * params_test$tau + sigma * rnorm(n)
  
  make_matrix = function(x) stats::model.matrix(~.-1, x)
  
  X_train = make_matrix(data.frame(params_train$X))
  X_test = make_matrix(data.frame(params_test$X))
  
  # causal forest honest split
  # estimate causal forest
  cf = causal_forest(X_train, Y_train, W_train,num.trees = ntree,min.node.size = min_node)
  # prediction causal forest
  cf.pred <- predict(cf, X_test, estimate.variance = TRUE)

  # se estimate causal forest
  se.hat = sqrt(cf.pred$variance.estimates)
  # cover rate
  cf.cov = abs(cf.pred$predictions - params_test$tau) <= 1.96 * se.hat
  cf.covered = mean(cf.cov)

  # mse
  cf.mse = mean((cf.pred$predictions - params_test$tau)^2)
  # bias
  cf.bias = mean(abs(cf.pred$predictions - params_test$tau))

  # same for causal forest adaptative split
  cf.adapt = causal_forest(X_train, Y_train, W_train,num.trees = ntree,min.node.size = min_node,honesty = F)
  cf.pred.adapt <- predict(cf.adapt, X_test, estimate.variance = TRUE)

  se.hat.adapt = sqrt(cf.pred.adapt$variance.estimates)
  cf.cov.adapt = abs(cf.pred.adapt$predictions - params_test$tau) <= 1.96 * se.hat.adapt
  cfadapt.covered = mean(cf.cov.adapt)

  cfadapt.mse = mean((cf.pred.adapt$predictions - params_test$tau)^2)
  cfadapt.bias = mean(abs(cf.pred.adapt$predictions - params_test$tau))

  # same for 10-nearest neighbour
  k.small = 10
  knn.small=causal.kn(kn=k.small,X_train,W_train,X_test,Y_train)
  knn.cov = abs(knn.small$knn.tau - params_test$tau) <= 1.96 * knn.small$knn.se
  knn.covered = mean(knn.cov)

  knn.mse = mean((knn.small$knn.tau - params_test$tau)^2)
  knn.bias = mean(abs(knn.small$knn.tau - params_test$tau))

  # same for 100-nearest neighbour
  k.big = 100
  knn.big=causal.kn(kn=k.big,X_train,W_train,X_test,Y_train)
  knnbig.cov = abs(knn.big$knn.tau - params_test$tau) <= 1.96 * knn.big$knn.se
  knnbig.covered = mean(knnbig.cov)

  knnbig.mse = mean((knn.big$knn.tau - params_test$tau)^2)
  knnbig.bias = mean(abs(knn.big$knn.tau - params_test$tau))

  # # same for inverse probability weight estimator (IPW) with lasso
  # ipwlasso.pred = ipw.lasso.estimator(X_test,Y_test,W_test)
  # ipwlasso.mse = mean((ipw.pred - params_test$tau)^2)
  # ipwlasso.bias = mean(abs(ipw.pred - params_test$tau))

  # same for inverse probability weight estimator (IPW) with random forest
  ipwrf.pred = ipw.rf.estimator(X_test,Y_test,W_test)
  ipwrf.mse = mean((ipwrf.pred - params_test$tau)^2)
  ipwrf.bias = mean(abs(ipwrf.pred - params_test$tau))

  # # same for augmented inverse probability weight estimator (AIPW) with lasso
  # aipwlasso.pred = aipw.lasso.estimator(X_test,Y_test,W_test)
  # aipwlasso.mse = mean((aipw.pred - params_test$tau)^2)
  # aipwlasso.bias = mean(abs(aipw.pred - params_test$tau))

  # same for augmented inverse probability weight estimator (AIPW) with random forest
  aipwrf.pred = aipw.rf.estimator(X_test,Y_test,W_test)
  aipwrf.mse = mean((aipwrf.pred - params_test$tau)^2)
  aipwrf.bias = mean(abs(aipwrf.pred - params_test$tau))
  
  # same for double/debiased machine learning (DML)
  dml.pred = dml.estimator(X=X_test,W=W_test,Y=Y_test)$tau_hat
  dml.mse = mean((dml.pred - params_test$tau)^2)
  dml.bias = mean(abs(dml.pred - params_test$tau))

  # same for double/debiased machine learning (DML) by DoubleML python package
  dados_train_full <- data.frame(Y=Y_train,W=W_train,X_train)
  cova_train=colnames(X_train)
  dados_train <- data.frame(X_train)
  dados_test <- data.frame(X_test)
  dmlpy_results <- dml_py(data_train_full = dados_train_full,
                  data_train = dados_train,
                  outcome = "Y",treatment = "W",
                  covariates = cova_train,data_test = dados_test,nboot = nbootdml)
  
  # lista <- c(200,400,600,1000,2000) %>% as.integer()
  # for(i in lista){
  #   start_time <- Sys.time()
  #   dmlpy_results <- dml_py(data_train_full = dados_train_full,
  #                           data_train = dados_train,
  #                           outcome = "Y",treatment = "W",
  #                           covariates = cova_train,data_test = dados_test,nboot = i)
  #   
  #   end_time <- Sys.time()
  #   time_taken <- difftime(end_time, start_time, units='mins')
  #   print(time_taken)  
  # }
  
  # prediction dml
  dmlpy.pred = dmlpy_results$effect
  
  # cover rate
  dmlpy.cov = dmlpy_results$effect >= dmlpy_results$`2.5 %` & 
              dmlpy_results$effect <= dmlpy_results$`97.5 %`
  dmlpy.covered = mean(dmlpy.cov)
  
  # mse
  dmlpy.mse = mean((dmlpy.pred - params_test$tau)^2)
  # bias
  dmlpy.bias = mean(abs(dmlpy.pred - params_test$tau))
  
  c(
    cf_covered = cf.covered,
    cf_mse = cf.mse,
    cf_bias = cf.bias,
    cfadapt_covered = cfadapt.covered,
    cfadapt_mse = cfadapt.mse,
    cfadapt_bias = cfadapt.bias,
    knn_covered = knn.covered,
    knn_mse = knn.mse,
    knn_bias = knn.bias,
    knnbig_covered = knnbig.covered,
    knnbig_mse = knnbig.mse,
    knnbig_bias = knnbig.bias,
    # ipwlasso_mse = ipwlasso.mse,
    # ipwlasso_bias = ipwlasso.bias,
    ipwrf_mse = ipwrf.mse,
    ipwrf_bias = ipwrf.bias,
    aipwrf_mse = aipwrf.mse,
    aipwrf_bias = aipwrf.bias,
    # aipwlasso_mse = aipwlasso.mse,
    # aipwlasso_bias = aipwlasso.bias,
    dml_mse = dml.mse,
    dml_bias = dml.bias,
    dmlpy_covered = dmlpy.covered,
    dmlpy_mse = dmlpy.mse,
    dmlpy_bias = dmlpy.bias
    )
  # function outputs
  # cover rate (covered), mean squared error (mse) and bias (bias)
  # for all estimation methods: 
  # causal forest, honest (cf) and adaptative split method (cfadapt)
  # k-nearest neighbor with a small (knn) and a big k (knnbig)
  # inverse probability weighting (ipw) and augmented (aipw)
  # double machine learning by R (dml) e by python (dmlpy)
}


# A. Set parameters -------------------------------------------------------

sigma=3;
# n=500;
d=20
# s = n/2 # sample size to honest partition method in causal forest
ntree = 400 # number of trees(B)
# min_node =  round(n*0.005) # minimum size node of tree
nbootdml = as.integer(1000) # number of bootstrap repetition
simu.reps = 500  #simulation replication
# sigma=c(0.5,1,3)
nvals= c(500)
# nvals= c(500,2000,5000)
# dvals=c(4,10,20)
setupvals=LETTERS[1:4]

# loop for all DGP --------------------------------------------------------

for(i in nvals){
  n=i
  s = n/2 # sample size to honest partition method in causal forest
  min_node =  round(n*0.005) # minimum size node of tree
  # n=500;d=4 
start_time <- Sys.time()
results.raw = lapply(setupvals, function(type) {
  print(paste("RUNNING FOR SETUP",type," WITH ",d, "COVARIATES, ",
              sigma, "SIGMA, ",
              n," SAMPLE SIZE."))
	res.d = sapply(1:simu.reps, function(iter) {
	  print(paste0("starting iteration ", iter))
	  simu.fun(n,d,sigma,setup = type)})
	# res.fixed = data.frame(t(res.d))
	# print(paste("RESULT AT", d, "IS", colMeans(res.fixed)))
	# res.fixed
	res.d
})
# fnm_rept = paste("results/saidarept", n, d, sigma,simu.reps,  "full.xlsx", sep="-")
# write_xlsx(results.raw,path = fnm_rept)

# results.condensed = rowMeans(data.frame(results.raw)) %>% as.data.frame() %>% 
#   rownames_to_column() %>% separate_wider_delim(rowname, 
#                                                 delim = "_",
#                                                 names = c("method", "measure")) %>% 
#   rename(value=".")

# results.condensed = lapply(results.raw, function(RR) {
#   RR.mu = colMeans(RR)
#   RR.var = sapply(RR, var) / (nrow(RR) - 1)
#   rbind("mu"=RR.mu, "se"=sqrt(RR.var))
#   })

results.condensed2 = lapply(results.raw, function(RR) {
  RR.mu = rowMeans(data.frame(RR)) %>% as.data.frame() %>% 
    rownames_to_column() %>% separate_wider_delim(rowname, 
                                                  delim = "_",
                                                  names = c("method", "measure")) %>% 
    rename(value=".")
    
    })
results.condensed <- map2(results.condensed2, setupvals, ~cbind(.x, setup = .y)) %>% 
  bind_rows()
end_time <- Sys.time()
time_taken <- difftime(end_time, start_time, units='mins')
results.condensed2 <- cbind(results.condensed,time_taken=as.numeric(time_taken),
      n, s, ntree, min_node, sigma,d,simu.reps)
fnm = paste("results/oficial", n, d, sigma,simu.reps,  "full.xlsx", sep="-")
write_xlsx(results.condensed2,path = fnm)
print(time_taken)
}

# B. construct tables --------------------------------------------------------

# read all worksheets for selected folder
end.temp <- "C:/Users/renat/Dropbox/Dissertacao-Renato/2.simulacoes de causalidade/results2106"
setwd(end.temp)
file.list <- list.files(pattern='*.xlsx')
alldata <- file.list %>%
  map_dfr(~read_excel(.x))

# list of methods excluded in table
exclusion_list <- c("knn","knnbig","dml")

# B1. Estimation MSE for n = 2000, d = 4, sigma = 3, for all setups (A, B, C, D) ---------------------------------------------------------------------
# filtering table Comparing RF – Honesty Sample & Adaptive, IPW, AIPW, DML
alldata %>% 
  filter(!method %in% exclusion_list,measure=="mse",
         n==2000,d==4,sigma==3) %>% 
  select(method,value,setup,n) %>% 
  mutate(value=round(value, 3),n=as.integer(n)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,n) -> dataselected
xtab = xtable(dataselected,
              caption = "MSE médio de 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, IPW, AIPW e DML) 
       com número de covariáveis d=4, tamanho de amostra n=2000 e sigma=3.",
              label = "tab:mse-cov4-n2000-sigma3",
              digits = 3,
              align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B1a1. Estimation MSE for n = 2000, d = 10, sigma = 3, for all setups (A, B, C, D) ---------------------------------------------------------------------
# filtering table Comparing RF – Honesty Sample & Adaptive, IPW, AIPW, DML 
alldata %>% 
  filter(!method %in% exclusion_list,measure=="mse",
         n==2000,d==10,sigma==3) %>% 
  select(method,value,setup,n) %>% 
  mutate(value=round(value, 3),n=as.integer(n)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,n) -> dataselected
xtab = xtable(dataselected,
              caption = "MSE médio de 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, IPW, AIPW e DML) 
       com número de covariáveis d=10, tamanho de amostra n=2000 e sigma=3.",
       label = "tab:mse-cov10-n2000-sigma3",
       digits = 3,
       align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B1a2. Estimation MSE for n = 2000, d = 20, sigma = 3, for all setups (A, B, C, D) ---------------------------------------------------------------------
# filtering table Comparing RF – Honesty Sample & Adaptive, IPW, AIPW, DML 
alldata %>% 
  filter(!method %in% exclusion_list,measure=="mse",
         n==2000,d==20,sigma==3) %>% 
  select(method,value,setup,n) %>% 
  mutate(value=round(value, 3),n=as.integer(n)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,n) -> dataselected
xtab = xtable(dataselected,
              caption = "MSE médio de 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, IPW, AIPW e DML) 
       com número de covariáveis d=20, tamanho de amostra n=2000 e sigma=3.",
       label = "tab:mse-cov20-n2000-sigma3",
       digits = 3,
       align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B1.2. Estimation Bias for n = 2000, d = 4, sigma = 3, for all setups (A, B, C, D) ---------------------------------------------------------------------
# filtering table Comparing RF – Honesty Sample & Adaptive, IPW, AIPW, DML 
alldata %>% 
  filter(!method %in% exclusion_list,measure=="bias",
         n==2000,d==4,sigma==3) %>% 
  select(method,value,setup,n) %>% 
  mutate(value=round(value, 3),n=as.integer(n)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,n) -> dataselected
xtab = xtable(dataselected,
              caption = "Viés absoluto para 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, IPW, AIPW e DML) 
       com número de covariáveis d=4, tamanho de amostra n=2000 e sigma=3.",
              label = "tab:bias-cov4-n2000-sigma3",
              digits = 3,
              align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B1.2a1. Estimation Bias for n = 2000, d = 10, sigma = 3, for all setups (A, B, C, D) ---------------------------------------------------------------------
# filtering table Comparing RF – Honesty Sample & Adaptive, IPW, AIPW, DML 
alldata %>% 
  filter(!method %in% exclusion_list,measure=="bias",
         n==2000,d==10,sigma==3) %>% 
  select(method,value,setup,n) %>% 
  mutate(value=round(value, 3),n=as.integer(n)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,n) -> dataselected
xtab = xtable(dataselected,
              caption = "Viés absoluto para 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, IPW, AIPW e DML) 
       com número de covariáveis d=10, tamanho de amostra n=2000 e sigma=3.",
              label = "tab:bias-cov10-n2000-sigma3",
              digits = 3,
              align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B1.2a2. Estimation Bias for n = 2000, d = 20, sigma = 3, for all setups (A, B, C, D) ---------------------------------------------------------------------
# filtering table Comparing RF – Honesty Sample & Adaptive, IPW, AIPW, DML 
alldata %>% 
  filter(!method %in% exclusion_list,measure=="bias",
         n==2000,d==20,sigma==3) %>% 
  select(method,value,setup,n) %>% 
  mutate(value=round(value, 3),n=as.integer(n)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,n) -> dataselected
xtab = xtable(dataselected,
              caption = "Viés absoluto para 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, IPW, AIPW e DML) 
       com número de covariáveis d=20, tamanho de amostra n=2000 e sigma=3.",
       label = "tab:bias-cov20-n2000-sigma3",
       digits = 3,
       align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B1.3. Covered Rate for n = 2000, d = 4, sigma = 3, for all setups (A, B, C, D) ---------------------------------------------------------------------

alldata %>% 
  filter(method %in% c("cf","cfadapt","knn","knnbig","dmlpy"),measure=="covered",
         n==2000,d==4,sigma==3) %>% 
  select(method,value,setup,n) %>% 
  mutate(value=round(value, 2),n=as.integer(n)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,n) -> dataselected
xtab = xtable(dataselected,
              caption = "Taxa de cobertura para 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, KNN-10, KNN-100 e DML) 
       com número de covariáveis d=4, tamanho de amostra n=2000 e sigma=3.",
       label = "tab:covered-cov4-n2000-sigma3",
       digits = 2,
       align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B1.3a1. Covered Rate for n = 2000, d = 10, sigma = 3, for all setups (A, B, C, D) ---------------------------------------------------------------------

alldata %>% 
  filter(method %in% c("cf","cfadapt","knn","knnbig","dmlpy"),measure=="covered",
         n==2000,d==10,sigma==3) %>% 
  select(method,value,setup,n) %>% 
  mutate(value=round(value, 2),n=as.integer(n)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,n) -> dataselected
xtab = xtable(dataselected,
              caption = "Taxa de cobertura para 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, KNN-10, KNN-100 e DML) 
       com número de covariáveis d=10, tamanho de amostra n=2000 e sigma=3.",
              label = "tab:covered-cov10-n2000-sigma3",
              digits = 2,
              align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B1.3a2. Covered Rate for n = 2000, d = 20, sigma = 3, for all setups (A, B, C, D) ---------------------------------------------------------------------

alldata %>% 
  filter(method %in% c("cf","cfadapt","knn","knnbig","dmlpy"),measure=="covered",
         n==2000,d==20,sigma==3) %>% 
  select(method,value,setup,n) %>% 
  mutate(value=round(value, 2),n=as.integer(n)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,n) -> dataselected
xtab = xtable(dataselected,
              caption = "Taxa de cobertura para 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, KNN-10, KNN-100 e DML) 
       com número de covariáveis d=20, tamanho de amostra n=2000 e sigma=3.",
       label = "tab:covered-cov20-n2000-sigma3",
       digits = 2,
       align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B2. Consistency MSE for d = 4, sigma = 3, for all setups (A, B, C, D) and n=500,2000,5000 ---------------------------------------------------------------------
alldata %>% 
  filter(!method %in% exclusion_list,measure=="mse",
         d==4,sigma==3) %>% 
  select(method,value,setup,n) %>% 
  mutate(value=round(value, 3),n=as.integer(n)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,n) -> dataselected
xtab = xtable(dataselected,
              caption = "MSE médio de 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, IPW, AIPW e DML) 
       por tamanho de amostra com número de covariáveis d=4 e sigma=3.",
              label = "tab:mse-por-amostra-cov4",
              digits = 3,
              align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B2a1. Consistency MSE for d = 10, sigma = 3, for all setups (A, B, C, D) and n=500,2000,5000 ---------------------------------------------------------------------
alldata %>% 
  filter(!method %in% exclusion_list,measure=="mse",
         d==10,sigma==3) %>% 
  select(method,value,setup,n) %>% 
  mutate(value=round(value, 3),n=as.integer(n)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,n) -> dataselected
xtab = xtable(dataselected,
              caption = "MSE médio de 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, IPW, AIPW e DML) 
       por tamanho de amostra com número de covariáveis d=10 e sigma=3.",
       label = "tab:mse-por-amostra-cov10",
       digits = 3,
       align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B2a2. Consistency MSE for d = 20, sigma = 3, for all setups (A, B, C, D) and n=500,2000,5000 ---------------------------------------------------------------------
alldata %>% 
  filter(!method %in% exclusion_list,measure=="mse",
         d==20,sigma==3) %>% 
  select(method,value,setup,n) %>% 
  mutate(value=round(value, 3),n=as.integer(n)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,n) -> dataselected
xtab = xtable(dataselected,
              caption = "MSE médio de 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, IPW, AIPW e DML) 
       por tamanho de amostra com número de covariáveis d=20 e sigma=3.",
       label = "tab:mse-por-amostra-cov20",
       digits = 3,
       align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B2.2. Consistency Bias for d = 4, sigma = 3, for all setups (A, B, C, D) and n=500,2000,5000 ---------------------------------------------------------------------
alldata %>% 
  filter(!method %in% exclusion_list, measure=="bias",
         d==4,sigma==3) %>% 
  select(method,value,setup,n) %>% 
  mutate(value=round(value, 3),n=as.integer(n)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,n) -> dataselected
xtab = xtable(dataselected,
              caption = "Viés absoluto de 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, IPW, AIPW e DML) 
       por tamanho de amostra com número de covariáveis d=4 e sigma=3.",
              label = "tab:bias-por-amostra-cov4",
              digits = 3,
              align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B2.2a1. Consistency Bias for d = 10, sigma = 3, for all setups (A, B, C, D) and n=500,2000,5000 ---------------------------------------------------------------------
alldata %>% 
  filter(!method %in% exclusion_list, measure=="bias",
         d==10,sigma==3) %>% 
  select(method,value,setup,n) %>% 
  mutate(value=round(value, 3),n=as.integer(n)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,n) -> dataselected
xtab = xtable(dataselected,
              caption = "Viés absoluto de 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, IPW, AIPW e DML) 
       por tamanho de amostra com número de covariáveis d=10 e sigma=3.",
       label = "tab:bias-por-amostra-cov10",
       digits = 3,
       align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B2.2a2. Consistency Bias for d = 20, sigma = 3, for all setups (A, B, C, D) and n=500,2000,5000 ---------------------------------------------------------------------
alldata %>% 
  filter(!method %in% exclusion_list, measure=="bias",
         d==20,sigma==3) %>% 
  select(method,value,setup,n) %>% 
  mutate(value=round(value, 3),n=as.integer(n)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,n) -> dataselected
xtab = xtable(dataselected,
              caption = "Viés absoluto de 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, IPW, AIPW e DML) 
       por tamanho de amostra com número de covariáveis d=20 e sigma=3.",
       label = "tab:bias-por-amostra-cov20",
       digits = 3,
       align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B3. Covered Rate for d = 4, sigma = 3, for all setups (A, B, C, D) and n=500,2000,5000 ---------------------------------------------------------------------
alldata %>% 
  filter(method %in% c("cf","cfadapt","knn","knnbig","dmlpy"),measure=="covered",
         d==4,sigma==3) %>% 
  select(method,value,setup,n) %>% 
  mutate(value=round(value, 2),n=as.integer(n)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,n) -> dataselected
xtab = xtable(dataselected,
              caption = "Taxa de cobertura de 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, KNN-10, KNN-100 e DML) 
       por tamanho de amostra com número de covariáveis d=4 e sigma=3.",
              label = "tab:covered-por-amostra-cov4",
              digits = 2,
              align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B3a1. Covered Rate for d = 10, sigma = 3, for all setups (A, B, C, D) and n=500,2000,5000 ---------------------------------------------------------------------
alldata %>% 
  filter(method %in% c("cf","cfadapt","knn","knnbig","dmlpy"),measure=="covered",
         d==10,sigma==3) %>% 
  select(method,value,setup,n) %>% 
  mutate(value=round(value, 2),n=as.integer(n)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,n) -> dataselected
xtab = xtable(dataselected,
              caption = "Taxa de cobertura de 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, KNN-10, KNN-100 e DML) 
       por tamanho de amostra com número de covariáveis d=10 e sigma=3.",
       label = "tab:covered-por-amostra-cov10",
       digits = 2,
       align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B3a2. COMPLETAR Covered Rate for d = 20, sigma = 3, for all setups (A, B, C, D) and n=500,2000,5000 ---------------------------------------------------------------------
alldata %>% 
  filter(method %in% c("cf","cfadapt","knn","knnbig","dmlpy"),measure=="covered",
         d==20,sigma==3) %>% 
  select(method,value,setup,n) %>% 
  mutate(value=round(value, 2),n=as.integer(n)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,n) -> dataselected
xtab = xtable(dataselected,
              caption = "Taxa de cobertura de 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, KNN-10, KNN-100 e DML) 
       por tamanho de amostra com número de covariáveis d=20 e sigma=3.",
       label = "tab:covered-por-amostra-cov20",
       digits = 2,
       align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B4. Dimensionality MSE for n = 2000, sigma = 3, for all setups (A, B, C, D) and d=4,10,20 ---------------------------------------------------------------------
alldata %>% 
  filter(!method %in% exclusion_list,measure=="mse",
         n==2000,sigma==3) %>% 
  select(method,value,setup,d) %>% 
  mutate(value=round(value, 3),d=as.integer(d)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,d) -> dataselected
xtab = xtable(dataselected,
              caption = "MSE médio de 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, IPW, AIPW e DML) 
       por número de covariáveis com tamanho de amostra n=2000 e sigma=3.",
              label = "tab:mse-por-cov-n2000",
              digits = 3,
              align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B4a1. Dimensionality MSE for n = 500, sigma = 3, for all setups (A, B, C, D) and d=4,10,20 ---------------------------------------------------------------------
alldata %>% 
  filter(!method %in% exclusion_list,measure=="mse",
         n==500,sigma==3) %>% 
  select(method,value,setup,d) %>% 
  mutate(value=round(value, 3),d=as.integer(d)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,d) -> dataselected
xtab = xtable(dataselected,
              caption = "MSE médio de 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, IPW, AIPW e DML) 
       por número de covariáveis com tamanho de amostra n=500 e sigma=3.",
       label = "tab:mse-por-cov-n500",
       digits = 3,
       align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B4a2. COMPLETAR Dimensionality MSE for n = 5000, sigma = 3, for all setups (A, B, C, D) and d=4,10,20 ---------------------------------------------------------------------
alldata %>% 
  filter(!method %in% exclusion_list,measure=="mse",
         n==5000,sigma==3) %>% 
  select(method,value,setup,d) %>% 
  mutate(value=round(value, 3),d=as.integer(d)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,d) -> dataselected
xtab = xtable(dataselected,
              caption = "MSE médio de 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, IPW, AIPW e DML) 
       por número de covariáveis com tamanho de amostra n=5000 e sigma=3.",
       label = "tab:mse-por-cov-n5000",
       digits = 3,
       align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B4.2. Dimensionality Bias for n = 2000, sigma = 3, for all setups (A, B, C, D) and d=4,10,20 ---------------------------------------------------------------------
alldata %>% 
  filter(!method %in% exclusion_list,measure=="bias",
         n==2000,sigma==3) %>% 
  select(method,value,setup,d) %>% 
  mutate(value=round(value, 3),d=as.integer(d)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,d) -> dataselected
xtab = xtable(dataselected,
              caption = "Viés absoluto para 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, IPW, AIPW e DML) 
       por número de covariáveis com tamanho de amostra n=2000 e sigma=3.",
              label = "tab:bias-por-cov-n2000",
              digits = 3,
              align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B4.2a1. Dimensionality Bias for n = 500, sigma = 3, for all setups (A, B, C, D) and d=4,10,20 ---------------------------------------------------------------------
alldata %>% 
  filter(!method %in% exclusion_list,measure=="bias",
         n==500,sigma==3) %>% 
  select(method,value,setup,d) %>% 
  mutate(value=round(value, 3),d=as.integer(d)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,d) -> dataselected
xtab = xtable(dataselected,
              caption = "Viés absoluto para 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, IPW, AIPW e DML) 
       por número de covariáveis com tamanho de amostra n=500 e sigma=3.",
       label = "tab:bias-por-cov-n500",
       digits = 3,
       align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B4.2a2. COMPLETAR Dimensionality Bias for n = 5000, sigma = 3, for all setups (A, B, C, D) and d=4,10,20 ---------------------------------------------------------------------
alldata %>% 
  filter(!method %in% exclusion_list,measure=="bias",
         n==5000,sigma==3) %>% 
  select(method,value,setup,d) %>% 
  mutate(value=round(value, 3),d=as.integer(d)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,d) -> dataselected
xtab = xtable(dataselected,
              caption = "Viés absoluto para 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, IPW, AIPW e DML) 
       por número de covariáveis com tamanho de amostra n=5000 e sigma=3.",
       label = "tab:bias-por-cov-n5000",
       digits = 3,
       align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B4.3. Dimensionality Covered for n = 2000, sigma = 3, for all setups (A, B, C, D) and d=4,10,20 ---------------------------------------------------------------------
alldata %>% 
  filter(method %in% c("cf","cfadapt","knn","knnbig","dmlpy"),measure=="covered",
         n==2000,sigma==3) %>% 
  select(method,value,setup,d) %>% 
  mutate(value=round(value, 2),d=as.integer(d)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,d) -> dataselected
xtab = xtable(dataselected,
              caption = "Taxa de cobertura para 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, KNN-10, KNN-100 e DML) 
       por número de covariáveis com tamanho de amostra n=2000 e sigma=3.",
              label = "tab:covered-por-cov-n2000",
              digits = 2,
              align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B4.3a1. Dimensionality Covered for n = 500, sigma = 3, for all setups (A, B, C, D) and d=4,10,20 ---------------------------------------------------------------------
alldata %>% 
  filter(method %in% c("cf","cfadapt","knn","knnbig","dmlpy"),measure=="covered",
         n==500,sigma==3) %>% 
  select(method,value,setup,d) %>% 
  mutate(value=round(value, 2),d=as.integer(d)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,d) -> dataselected
xtab = xtable(dataselected,
              caption = "Taxa de cobertura para 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, KNN-10, KNN-100 e DML) 
       por número de covariáveis com tamanho de amostra n=500 e sigma=3.",
       label = "tab:covered-por-cov-n500",
       digits = 2,
       align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)

# B4.3a2. COMPLETAR Dimensionality Covered for n = 5000, sigma = 3, for all setups (A, B, C, D) and d=4,10,20 ---------------------------------------------------------------------
alldata %>% 
  filter(method %in% c("cf","cfadapt","knn","knnbig","dmlpy"),measure=="covered",
         n==5000,sigma==3) %>% 
  select(method,value,setup,d) %>% 
  mutate(value=round(value, 2),d=as.integer(d)) %>% 
  pivot_wider(names_from = method ,values_from = value
  ) %>% 
  arrange(setup,d) -> dataselected
xtab = xtable(dataselected,
              caption = "Taxa de cobertura para 500 replicações dos métodos estudados (Floresta Aleatória -- Amostra Honesta e Adaptativa, KNN-10, KNN-100 e DML) 
       por número de covariáveis com tamanho de amostra n=5000 e sigma=3.",
       label = "tab:covered-por-cov-n5000",
       digits = 2,
       align = "rlr|rrrr|r")
print(xtab, include.rownames = FALSE)


# 4. Plots ----------------------------------------------------------------

setupvals=LETTERS[1:4]
ntree = 400 # number of trees(B)
nbootdml = as.integer(1000) # number of bootstrap repetition

# function to create a X1 variable grid and other constants
# also provide true tau (CATE) and estimated values for all methods
# on X defined vector of variables.
params.x1.setup = function(n=500,d=4,sigma=3,eta = 0.1,setup = "A") {
  s = n/2 # sample size to honest partition method in causal forest
  min_node =  round(n*0.005) # minimum size node of tree
  # function to choose the data generating process (DGP) parameters
  # the output is a list with X=X, b=b, tau=tau, e=e parameters
  if (setup == 'A') {#linear setup
    get.params = function() {
      X = matrix(runif(n * d, min=0, max=1), n, d) # covariates
      b = X[,1] + 0.5*X[,2] + 0.4*X[,3] # baseline main effect
      tau = 0.3*X[,1] + 0.6*X[,2] + 0.6*X[,3] + 2.2*X[,4] # effect = tau(CATE)
      eta = 0.1
      e = pmax(eta, pmin(sin(pi * X[,1] * X[,2]), 1-eta)) #propensity score trimmed eta 0.1
      list(X=X, b=b, tau=tau, e=e)
    }
  } else if (setup == 'B') {# nonlinear setup
    get.params = function() {
      X = matrix(runif(n*d, min=0, max=1), n, d) # covariates
      b = 0.2*X[,1] + X[,1]^2 + 0.5*X[,2]*X[,3] + 0.4*X[,3] + 0.8*X[,3]^2 # baseline main effect
      eta = 0.1
      e = pmax(eta, pmin(X[,1]- 0.2*X[,2]^2, 1-eta)) #propensity score trimmed eta 0.1
      tau = 0.3*X[,1] + 0.6*X[,2] + 0.6*X[,3] + 2.2*X[,4]  # effect = tau(CATE)
      list(X=X, b=b, tau=tau, e=e)
    }
  } else if (setup == 'C') {# peaks and valleys setup
    get.params = function() {
      X = matrix(runif(n * d, min=0, max=1), n, d) # covariates
      b = 0 # baseline main effect
      tau = (1 + 1 / (1 + exp(-20 * (X[,1] - 1/3)))) * (1 + 1 / (1 + exp(-20 * (X[,2] - 1/3))))* (1 + 1 / (1 + exp(-20 * (X[,3] - 1/3)))) # effect = tau(CATE)
      e = pmax(eta, pmin(0.5*X[,1] + 0.5*X[,2], 1-eta)) #propensity score trimmed eta 0.1
      list(X=X, b=b, tau=tau, e=e)
    }
  } else if (setup == 'D') {# discontinuities setup
    get.params = function() {
      X = matrix(rnorm(n * d), n, d) # covariates
      b = 2 * (X[,1]>0.4) + 0.3*X[,2] # baseline main effect
      eta = 0.1
      e = pmax(eta, pmin(0.7*X[,1]- 0.7*X[,2]+ 0.7*X[,3], 1-eta)) #propensity score trimmed eta 0.1
      tau = 2 * (X[,1]>0.6) + 1.5 * (X[,2]>0.6) + 0.3*X[,3] - 0.7*X[,4]  # effect = tau(CATE)
      list(X=X, b=b, tau=tau, e=e)
    }
  } else {
    
    stop("bad setup")
  }
  
  # function to choose the data generating process (DGP) parameters
  # the output is a list with X=X, b=b, tau=tau, e=e parameters
  if (setup == 'A') {#linear setup
    get.params.t = function() {
      X.grid=matrix(c(seq(-1, 1, by= 0.01),rep(0.5,times=201*(d-1))),length(seq(-1, 1, by= 0.01)),d)
      X = X.grid
      b = X[,1] + 0.5*X[,2] + 0.4*X[,3] # baseline main effect
      tau = 0.3*X[,1] + 0.6*X[,2] + 0.6*X[,3] + 2.2*X[,4] # effect = tau(CATE)
      eta = 0.1
      e = pmax(eta, pmin(sin(pi * X[,1] * X[,2]), 1-eta)) #propensity score trimmed eta 0.1
      list(X=X, b=b, tau=tau, e=e)
    }
  } else if (setup == 'B') {# nonlinear setup
    get.params.t = function() {
      X.grid=matrix(c(seq(-1, 1, by= 0.01),rep(0.5,times=201*(d-1))),length(seq(-1, 1, by= 0.01)),d)
      X = X.grid
      b = 0.2*X[,1] + X[,1]^2 + 0.5*X[,2]*X[,3] + 0.4*X[,3] + 0.8*X[,3]^2 # baseline main effect
      eta = 0.1
      e = pmax(eta, pmin(X[,1]- 0.2*X[,2]^2, 1-eta)) #propensity score trimmed eta 0.1
      tau = 0.3*X[,1] + 0.6*X[,2] + 0.6*X[,3] + 2.2*X[,4]  # effect = tau(CATE)
      list(X=X, b=b, tau=tau, e=e)
    }
  } else if (setup == 'C') {# peaks and valleys setup
    get.params.t = function() {
      X.grid=matrix(c(seq(-1, 1, by= 0.01),rep(0.5,times=201*(d-1))),length(seq(-1, 1, by= 0.01)),d)
      X = X.grid
      b = 0 # baseline main effect
      tau = (1 + 1 / (1 + exp(-20 * (X[,1] - 1/3)))) * (1 + 1 / (1 + exp(-20 * (X[,2] - 1/3))))* (1 + 1 / (1 + exp(-20 * (X[,3] - 1/3)))) # effect = tau(CATE)
      e = pmax(eta, pmin(0.5*X[,1] + 0.5*X[,2], 1-eta)) #propensity score trimmed eta 0.1
      list(X=X, b=b, tau=tau, e=e)
    }
  } else if (setup == 'D') {# discontinuities setup
    get.params.t = function() {
      X.grid=matrix(c(seq(-1, 1, by= 0.01),rep(0.5,times=201*(d-1))),length(seq(-1, 1, by= 0.01)),d)
      X = X.grid
      b = 2 * (X[,1]>0.4) + 0.3*X[,2] # baseline main effect
      eta = 0.1
      e = pmax(eta, pmin(0.7*X[,1]- 0.7*X[,2]+ 0.7*X[,3], 1-eta)) #propensity score trimmed eta 0.1
      tau = 2 * (X[,1]>0.6) + 1.5 * (X[,2]>0.6) + 0.3*X[,3] - 0.7*X[,4]  # effect = tau(CATE)
      list(X=X, b=b, tau=tau, e=e)
    }
  } else {
    
    stop("bad setup")
  }
  params_train = get.params()
  W_train = rbinom(n, 1, params_train$e)
  Y_train = params_train$b + (W_train - 0.5) * params_train$tau + sigma * rnorm(n)
  
  params_test = get.params.t()
  n2 <- length(params_test$e)
  W_test = rbinom(n2, 1, params_test$e)
  Y_test = params_test$b + (W_test - 0.5) * params_test$tau
  
  make_matrix = function(x) stats::model.matrix(~.-1, x)
  
  X_train = make_matrix(data.frame(params_train$X))
  X_test = make_matrix(data.frame(params_test$X))
  
  # causal forest honest split
  # estimate causal forest
  cf = causal_forest(X_train, Y_train, W_train,num.trees = ntree,min.node.size = min_node)
  # prediction causal forest
  cf.pred <- predict(cf, X_test, estimate.variance = TRUE)
  
  # se estimate causal forest
  se.hat = sqrt(cf.pred$variance.estimates)
  
  # same for causal forest adaptative split
  cf.adapt = causal_forest(X_train, Y_train, W_train,num.trees = ntree,min.node.size = min_node,honesty = F)
  cf.pred.adapt <- predict(cf.adapt, X_test, estimate.variance = TRUE)
  se.hat.adapt = sqrt(cf.pred.adapt$variance.estimates)
  
  # same for 10-nearest neighbour
  k.small = 10
  knn.small=causal.kn(kn=k.small,X_train,W_train,X_test,Y_train)
  se.hat.kkn.small= knn.small$knn.se
  
  # same for 100-nearest neighbour
  k.big = 100
  knn.big=causal.kn(kn=k.big,X_train,W_train,X_test,Y_train)
  se.hat.kkn.small= knn.big$knn.se
  
  # same for inverse probability weight estimator (IPW) with random forest
  ipwrf.pred = ipw.rf.estimator(X_test,Y_test,W_test)
  
  # same for augmented inverse probability weight estimator (AIPW) with random forest
  aipwrf.pred = aipw.rf.estimator(X_test,Y_test,W_test)
  
  # same for double/debiased machine learning (DML)
  dml.pred = dml.estimator(X=X_test,W=W_test,Y=Y_test)$tau_hat
  
  # same for double/debiased machine learning (DML) by DoubleML python package
  dados_train_full <- data.frame(Y=Y_train,W=W_train,X_train)
  cova_train=colnames(X_train)
  dados_train <- data.frame(X_train)
  dados_test <- data.frame(X_test)
  dmlpy_results <- dml_py(data_train_full = dados_train_full,
                          data_train = dados_train,
                          outcome = "Y",treatment = "W",
                          covariates = cova_train,data_test = dados_test,nboot = nbootdml)
  
  # prediction dml
  dmlpy.pred = dmlpy_results$effect
  
  results <- list(
    params_test=params_test,
    predtruth = list(tau = params_test$tau,cf = cf.pred$predictions,
                     cfadpt=cf.pred.adapt$predictions,dml = dmlpy.pred,
                     ipw=ipwrf.pred,aipw=aipwrf.pred)
    # W_test=W_test,X_test=X_test,Y_test=Y_test,
    # cf.pred=cf.pred,se.hat=se.hat,
    # cf.pred.adapt=cf.pred.adapt,se.hat.adapt=se.hat.adapt,
    # knn.small=knn.small,se.hat.kkn.small=se.hat.kkn.small,
    # knn.big=knn.big,se.hat.kkn.small=se.hat.kkn.small,
    # ipwrf.pred=ipwrf.pred,aipwrf.pred=aipwrf.pred,
    # dml.pred=dml.pred
    # dmlpy.pred=dmlpy.pred
  )
  results
}

# function to create a X2 variable grid and other constants
# also provide true tau (CATE) and estimated values for all methods
# on X defined vector of variables.
params.x2.setup = function(n=500,d=4,sigma=3,eta = 0.1,setup = "A") {
  # function to choose the data generating process (DGP) parameters
  # the output is a list with X=X, b=b, tau=tau, e=e parameters
  s = n/2 # sample size to honest partition method in causal forest
  min_node =  round(n*0.005) # minimum size node of tree
  if (setup == 'A') {#linear setup
    get.params = function() {
      X = matrix(runif(n * d, min=0, max=1), n, d) # covariates
      b = X[,1] + 0.5*X[,2] + 0.4*X[,3] # baseline main effect
      tau = 0.3*X[,1] + 0.6*X[,2] + 0.6*X[,3] + 2.2*X[,4] # effect = tau(CATE)
      eta = 0.1
      e = pmax(eta, pmin(sin(pi * X[,1] * X[,2]), 1-eta)) #propensity score trimmed eta 0.1
      list(X=X, b=b, tau=tau, e=e)
    }
  } else if (setup == 'B') {# nonlinear setup
    get.params = function() {
      X = matrix(runif(n*d, min=0, max=1), n, d) # covariates
      b = 0.2*X[,1] + X[,1]^2 + 0.5*X[,2]*X[,3] + 0.4*X[,3] + 0.8*X[,3]^2 # baseline main effect
      eta = 0.1
      e = pmax(eta, pmin(X[,1]- 0.2*X[,2]^2, 1-eta)) #propensity score trimmed eta 0.1
      tau = 0.3*X[,1] + 0.6*X[,2] + 0.6*X[,3] + 2.2*X[,4]  # effect = tau(CATE)
      list(X=X, b=b, tau=tau, e=e)
    }
  } else if (setup == 'C') {# peaks and valleys setup
    get.params = function() {
      X = matrix(runif(n * d, min=0, max=1), n, d) # covariates
      b = 0 # baseline main effect
      tau = (1 + 1 / (1 + exp(-20 * (X[,1] - 1/3)))) * (1 + 1 / (1 + exp(-20 * (X[,2] - 1/3))))* (1 + 1 / (1 + exp(-20 * (X[,3] - 1/3)))) # effect = tau(CATE)
      e = pmax(eta, pmin(0.5*X[,1] + 0.5*X[,2], 1-eta)) #propensity score trimmed eta 0.1
      list(X=X, b=b, tau=tau, e=e)
    }
  } else if (setup == 'D') {# discontinuities setup
    get.params = function() {
      X = matrix(rnorm(n * d), n, d) # covariates
      b = 2 * (X[,1]>0.4) + 0.3*X[,2] # baseline main effect
      eta = 0.1
      e = pmax(eta, pmin(0.7*X[,1]- 0.7*X[,2]+ 0.7*X[,3], 1-eta)) #propensity score trimmed eta 0.1
      tau = 2 * (X[,1]>0.6) + 1.5 * (X[,2]>0.6) + 0.3*X[,3] - 0.7*X[,4]  # effect = tau(CATE)
      list(X=X, b=b, tau=tau, e=e)
    }
  } else {
    
    stop("bad setup")
  }
  
  # function to choose the data generating process (DGP) parameters
  # the output is a list with X=X, b=b, tau=tau, e=e parameters
  if (setup == 'A') {#linear setup
    get.params.t = function() {
      X.grid=matrix(c(seq(-1, 1, by= 0.01),rep(0.5,times=201*(d-1))),length(seq(-1, 1, by= 0.01)),d)
      X = X.grid
      b = X[,1] + 0.5*X[,2] + 0.4*X[,3] # baseline main effect
      tau = 0.3*X[,1] + 0.6*X[,2] + 0.6*X[,3] + 2.2*X[,4] # effect = tau(CATE)
      eta = 0.1
      e = pmax(eta, pmin(sin(pi * X[,1] * X[,2]), 1-eta)) #propensity score trimmed eta 0.1
      list(X=X, b=b, tau=tau, e=e)
    }
  } else if (setup == 'B') {# nonlinear setup
    get.params.t = function() {
      X.grid=matrix(c(seq(-1, 1, by= 0.01),rep(0.5,times=201*(d-1))),length(seq(-1, 1, by= 0.01)),d)
      X = X.grid
      b = 0.2*X[,1] + X[,1]^2 + 0.5*X[,2]*X[,3] + 0.4*X[,3] + 0.8*X[,3]^2 # baseline main effect
      eta = 0.1
      e = pmax(eta, pmin(X[,1]- 0.2*X[,2]^2, 1-eta)) #propensity score trimmed eta 0.1
      tau = 0.3*X[,1] + 0.6*X[,2] + 0.6*X[,3] + 2.2*X[,4]  # effect = tau(CATE)
      list(X=X, b=b, tau=tau, e=e)
    }
  } else if (setup == 'C') {# peaks and valleys setup
    get.params.t = function() {
      X.grid=matrix(c(seq(-1, 1, by= 0.01),rep(0.5,times=201*(d-1))),length(seq(-1, 1, by= 0.01)),d)
      X = X.grid
      b = 0 # baseline main effect
      tau = (1 + 1 / (1 + exp(-20 * (X[,1] - 1/3)))) * (1 + 1 / (1 + exp(-20 * (X[,2] - 1/3))))* (1 + 1 / (1 + exp(-20 * (X[,3] - 1/3)))) # effect = tau(CATE)
      e = pmax(eta, pmin(0.5*X[,1] + 0.5*X[,2], 1-eta)) #propensity score trimmed eta 0.1
      list(X=X, b=b, tau=tau, e=e)
    }
  } else if (setup == 'D') {# discontinuities setup
    get.params.t = function() {
      X.grid=matrix(c(seq(-1, 1, by= 0.01),rep(0.5,times=201*(d-1))),length(seq(-1, 1, by= 0.01)),d)
      X = X.grid
      b = 2 * (X[,1]>0.4) + 0.3*X[,2] # baseline main effect
      eta = 0.1
      e = pmax(eta, pmin(0.7*X[,1]- 0.7*X[,2]+ 0.7*X[,3], 1-eta)) #propensity score trimmed eta 0.1
      tau = 2 * (X[,1]>0.6) + 1.5 * (X[,2]>0.6) + 0.3*X[,3] - 0.7*X[,4]  # effect = tau(CATE)
      list(X=X, b=b, tau=tau, e=e)
    }
  } else {
    
    stop("bad setup")
  }
  
  # function to choose the data generating process (DGP) parameters
  # the output is a list with X=X, b=b, tau=tau, e=e parameters
  if (setup == 'A') {#linear setup
    get.params.t = function() {
      X.grid=matrix(c(rep(0.5,times=201),seq(-1, 1, by= 0.01),rep(0.5,times=201*(d-2))),length(seq(-1, 1, by= 0.01)),d)
      X = X.grid
      b = X[,1] + 0.5*X[,2] + 0.4*X[,3] # baseline main effect
      tau = 0.3*X[,1] + 0.6*X[,2] + 0.6*X[,3] + 2.2*X[,4] # effect = tau(CATE)
      eta = 0.1
      e = pmax(eta, pmin(sin(pi * X[,1] * X[,2]), 1-eta)) #propensity score trimmed eta 0.1
      list(X=X, b=b, tau=tau, e=e)
    }
  } else if (setup == 'B') {# nonlinear setup
    get.params.t = function() {
      X.grid=matrix(c(rep(0.5,times=201),seq(-1, 1, by= 0.01),rep(0.5,times=201*(d-2))),length(seq(-1, 1, by= 0.01)),d)
      X = X.grid
      b = 0.2*X[,1] + X[,1]^2 + 0.5*X[,2]*X[,3] + 0.4*X[,3] + 0.8*X[,3]^2 # baseline main effect
      eta = 0.1
      e = pmax(eta, pmin(X[,1]- 0.2*X[,2]^2, 1-eta)) #propensity score trimmed eta 0.1
      tau = 0.3*X[,1] + 0.6*X[,2] + 0.6*X[,3] + 2.2*X[,4]  # effect = tau(CATE)
      list(X=X, b=b, tau=tau, e=e)
    }
  } else if (setup == 'C') {# peaks and valleys setup
    get.params.t = function() {
      X.grid=matrix(c(rep(0.5,times=201),seq(-1, 1, by= 0.01),rep(0.5,times=201*(d-2))),length(seq(-1, 1, by= 0.01)),d)
      X = X.grid
      b = 0 # baseline main effect
      tau = (1 + 1 / (1 + exp(-20 * (X[,1] - 1/3)))) * (1 + 1 / (1 + exp(-20 * (X[,2] - 1/3))))* (1 + 1 / (1 + exp(-20 * (X[,3] - 1/3)))) # effect = tau(CATE)
      e = pmax(eta, pmin(0.5*X[,1] + 0.5*X[,2], 1-eta)) #propensity score trimmed eta 0.1
      list(X=X, b=b, tau=tau, e=e)
    }
  } else if (setup == 'D') {# discontinuities setup
    get.params.t = function() {
      X.grid=matrix(c(rep(0.5,times=201),seq(-1, 1, by= 0.01),rep(0.5,times=201*(d-2))),length(seq(-1, 1, by= 0.01)),d)
      X = X.grid
      b = 2 * (X[,1]>0.4) + 0.3*X[,2] # baseline main effect
      eta = 0.1
      e = pmax(eta, pmin(0.7*X[,1]- 0.7*X[,2]+ 0.7*X[,3], 1-eta)) #propensity score trimmed eta 0.1
      tau = 2 * (X[,1]>0.6) + 1.5 * (X[,2]>0.6) + 0.3*X[,3] - 0.7*X[,4]  # effect = tau(CATE)
      list(X=X, b=b, tau=tau, e=e)
    }
  } else {
    
    stop("bad setup")
  }
  params_train = get.params()
  W_train = rbinom(n, 1, params_train$e)
  Y_train = params_train$b + (W_train - 0.5) * params_train$tau + sigma * rnorm(n)
  
  params_test = get.params.t()
  n2 <- dim(params_test$X)[1]
  W_test = rbinom(n2, 1, params_test$e)
  Y_test = params_test$b + (W_test - 0.5) * params_test$tau
  
  make_matrix = function(x) stats::model.matrix(~.-1, x)
  
  X_train = make_matrix(data.frame(params_train$X))
  X_test = make_matrix(data.frame(params_test$X))
  
  # causal forest honest split
  # estimate causal forest
  cf = causal_forest(X_train, Y_train, W_train,num.trees = ntree,min.node.size = min_node)
  # prediction causal forest
  cf.pred <- predict(cf, X_test, estimate.variance = TRUE)
  
  # se estimate causal forest
  se.hat = sqrt(cf.pred$variance.estimates)
  
  # same for causal forest adaptative split
  cf.adapt = causal_forest(X_train, Y_train, W_train,num.trees = ntree,min.node.size = min_node,honesty = F)
  cf.pred.adapt <- predict(cf.adapt, X_test, estimate.variance = TRUE)
  se.hat.adapt = sqrt(cf.pred.adapt$variance.estimates)
  
  # same for 10-nearest neighbour
  k.small = 10
  knn.small=causal.kn(kn=k.small,X_train,W_train,X_test,Y_train)
  se.hat.kkn.small= knn.small$knn.se
  
  # same for 100-nearest neighbour
  k.big = 100
  knn.big=causal.kn(kn=k.big,X_train,W_train,X_test,Y_train)
  se.hat.kkn.small= knn.big$knn.se
  
  # same for inverse probability weight estimator (IPW) with random forest
  ipwrf.pred = ipw.rf.estimator(X_test,Y_test,W_test)
  
  # same for augmented inverse probability weight estimator (AIPW) with random forest
  aipwrf.pred = aipw.rf.estimator(X_test,Y_test,W_test)
  
  # same for double/debiased machine learning (DML)
  dml.pred = dml.estimator(X=X_test,W=W_test,Y=Y_test)$tau_hat
  
  # same for double/debiased machine learning (DML) by DoubleML python package
  dados_train_full <- data.frame(Y=Y_train,W=W_train,X_train)
  cova_train=colnames(X_train)
  dados_train <- data.frame(X_train)
  dados_test <- data.frame(X_test)
  dmlpy_results <- dml_py(data_train_full = dados_train_full,
                          data_train = dados_train,
                          outcome = "Y",treatment = "W",
                          covariates = cova_train,data_test = dados_test,nboot = nbootdml)
  
  # prediction dml
  dmlpy.pred = dmlpy_results$effect
  
  
  results <- list(
    params_test=params_test,
    predtruth = list(tau = params_test$tau,cf = cf.pred$predictions,
                     cfadpt=cf.pred.adapt$predictions,dml = dmlpy.pred,
                     ipw=ipwrf.pred,aipw=aipwrf.pred)
    # W_test=W_test,X_test=X_test,Y_test=Y_test,
    # cf.pred=cf.pred,se.hat=se.hat,
    # cf.pred.adapt=cf.pred.adapt,se.hat.adapt=se.hat.adapt,
    # knn.small=knn.small,se.hat.kkn.small=se.hat.kkn.small,
    # knn.big=knn.big,se.hat.kkn.small=se.hat.kkn.small,
    # ipwrf.pred=ipwrf.pred,aipwrf.pred=aipwrf.pred,
    # dml.pred=dml.pred
    # dmlpy.pred=dmlpy.pred
  )
  results
}

# tt <- params.x1.setup(n=500,d=4,sigma=3,setup = "D")
# data1 <- data.frame(tt$params_test$X,
#                     tau=tt$params_test$tau)
# ggplot(data1)+
#   geom_line(aes(x=X1, y=tau)) + scale_y_continuous(
#     limits =   c(-2,10),
#     expand = expansion(mult = c(0,0.05)))
# 
# ss <- params.x2.setup(n=500,d=4,sigma=3,setup = "D")
# data2 <- data.frame(ss$params_test$X,
#                     tau=ss$params_test$tau)
# ggplot(data2) +
#   geom_line(aes(x=X2, y=tau))+ scale_y_continuous(
#     limits =   c(-2,10),
#     expand = expansion(mult = c(0,0.05)))


# 4A. true tau for X1 -----------------------------------------------------
tautruth <- lapply(setupvals, function(type) {
  tt <- params.x1.setup(n=500,d=4,sigma=3,setup = type)
  data1 <- data.frame(tt$params_test$X,
                      tau=tt$params_test$tau,setup = type)
  data1
  })
tautruth2 <- do.call(rbind.data.frame, tautruth)

ggplot(tautruth2) +
  geom_line(aes(x=X1, y=tau,color = setup),size=1.5,show.legend = FALSE)+
  facet_wrap(~setup)+
  labs(title = "Tau verdadeiro em diversos cenários do processo gerador para X1",
       subtitle = "Especificações: n=500,d=4,sigma=3, X1=[-1,1].") +
  theme_bw()

# 4B. true tau for X2 -----------------------------------------------------
tautruth3 <- lapply(setupvals, function(type) {
  tt <- params.x2.setup(n=500,d=4,sigma=3,setup = type)
  data1 <- data.frame(tt$params_test$X,
                      tau=tt$params_test$tau,setup = type)
  data1
})
tautruth4 <- do.call(rbind.data.frame, tautruth3)

ggplot(tautruth4)+
  geom_line(aes(x=X2, y=tau,color = setup),size=1.5,show.legend = FALSE)+
  facet_wrap(~setup)+
  labs(title = "Tau verdadeiro em diversos cenários do processo gerador para X2",
       subtitle = "Especificações: n=500,d=4,sigma=3, X2=[-1,1].") +
  theme_bw()

# 4C. Comparing estimated tau CF and DML for X1 ----------------------------------
comparetauX1 <- lapply(setupvals, function(type) {
  tt <- params.x1.setup(n=500,d=4,sigma=3,setup = type)
  data1 <- data.frame(tt$predtruth %>% as.data.frame(),
                      X1=tt$params_test$X[,1],
                      setup = type)
  data1
})
comparetauX12 <- do.call(rbind.data.frame, comparetauX1)
ggplot(comparetauX12, aes(x=X1))+
  geom_line(aes(y=tau,color="Tau"),size=1.5)+
  geom_line(aes(y=cf,color="CF"),size=1.5)+
  geom_line(aes(y=dml,color="DML"),size=1.5)+
  # geom_line(aes(y=ipw,color="IPW"),size=1.5)+
  # geom_line(aes(y=aipw,color="AIPW"),size=1.5)+
  facet_wrap(~setup,ncol = 1)+
  labs(color = "",y = "tau",
       title = "Comparação entre tau verdadeiro e valores estimados para X1",
       subtitle = "Floresta Causal X DML, cenários selecionados.") +
  theme_bw()

# 4D. Comparing estimated tau CF and DML for X2 ----------------------------------
comparetauX2 <- lapply(setupvals, function(type) {
  tt <- params.x2.setup(n=500,d=4,sigma=3,setup = type)
  data1 <- data.frame(tt$predtruth %>% as.data.frame(),
                      X2=tt$params_test$X[,2],
                      setup = type)
  data1
})
comparetauX22 <- do.call(rbind.data.frame, comparetauX2)
ggplot(comparetauX22, aes(x=X2))+
  geom_line(aes(y=tau,color="Tau"),size=1.5)+
  geom_line(aes(y=cf,color="CF"),size=1.5)+
  geom_line(aes(y=dml,color="DML"),size=1.5)+
  # geom_line(aes(y=ipw,color="IPW"),size=1.5)+
  # geom_line(aes(y=aipw,color="AIPW"),size=1.5)+
  facet_wrap(~setup)+
  labs(color = "",y = "tau",
       title = "Comparação entre tau verdadeiro e valores estimados para X2",
       subtitle = "Floresta Causal X DML, cenários selecionados.") +
  theme_bw()

# X.grid=matrix(rep(seq(-1, 1, by= 0.01),times=d),length(seq(-1, 1, by= 0.01)),d)
# te = apply(X.grid, 1, effect) # true effect for test sample
# cf.estCI <- predict(cf, X.grid, estimate.variance = TRUE)
# cf.adapt.estCI <- predict(cf.adapt, X.grid, estimate.variance = TRUE)
# knn.small.estCI=causal.kn(kn=k.small,X.grid,Y)
# knn.big.estCI=causal.kn(kn=k.big,X.grid,Y)
# 
# data1 = cbind(X=X.grid[,1],truth=te,cf=cf.estCI$predictions,
#               cf.adapt=cf.adapt.estCI$predictions,
#               knn.small=knn.small.estCI$knn.tau,
#               knn.big=knn.big.estCI$knn.tau) %>% as.data.frame
# 
# data1_long <- data1 %>% 
#   pivot_longer(cols = truth:knn.big, names_to = "grp",
#                values_to = "mean")


# comparing methods 
# (p1 <- ggplot(data1_long, aes(x=X, y=mean, color = grp))+
#     geom_line(aes(x=X, y=mean, color=grp)) +
#     # scale_color_viridis_d()+
#     labs(color = "",y = "tau") +
#     theme_bw()
# )
# 
# data2 = cbind(X=X.grid[,1],truth=te,cf=cf.estCI$predictions,
#               cf.se=sqrt(cf.estCI$variance.estimates)) %>% as.data.frame
# 
# head(data2)
# data2 <-  mutate(data2,
#                  upper.ci=cf+1.96*cf.se,
#                  lower.ci=cf-1.96*cf.se,
#                  cf.se=NULL)
# 
# # confidence interval band
# (p2 <- ggplot(data2, aes(x=X))+
#     # geom_point()+
#     geom_line(aes(y=truth,color="truth"))+
#     geom_ribbon(aes(ymin=lower.ci,ymax=upper.ci),alpha=0.3)+
#     geom_line(aes(y=cf,color="cf"))+
#     labs(color = "",y = "tau") +
#     theme_bw()
# )
# 
# 
# 
# 
# # Concatenate the two results.
# res <- rbind(forest.ate, ols.ate)
# 
# # Plotting the point estimate of average treatment effect 
# # and 95% confidence intervals around it.
# ggplot(res) +
#   aes(x = ranking, y = estimate, group=method, color=method) + 
#   geom_point(position=position_dodge(0.2)) +
#   geom_errorbar(aes(ymin=estimate-2*std.err, ymax=estimate+2*std.err), width=.2, position=position_dodge(0.2)) +
#   ylab("") + xlab("") +
#   ggtitle("Average CATE within each ranking (as defined by predicted CATE)") +
#   theme_minimal() +
#   theme(legend.position="bottom", legend.title = element_blank())
# 
# 
# # plot partial dependence
# selected.covariate <- "polviews"
# other.covariates <- covariates[which(covariates != selected.covariate)]
# 
# # Fitting a forest 
# # (commented for convenience; no need re-fit if already fitted above)
# fmla <- formula(paste0("~ 0 + ", paste0(covariates, collapse="+")))
# # Note: For smaller confidence intervals, set num.trees ~ sample size
# # X <- model.matrix(fmla, data)
# # W <- data[,treatment]
# # Y <- data[,outcome]
# # forest.tau <- causal_forest(X, Y, W, W.hat=.5)  # few trees for speed here
# 
# # Compute a grid of values appropriate for the selected covariate
# grid.size <- 7 
# covariate.grid <- seq(min(data[,selected.covariate]), max(data[,selected.covariate]), length.out=grid.size)
# 
# # Other options for constructing a grid:
# # For a binary variable, simply use 0 and 1
# # grid.size <- 2
# # covariate.grid <- c(0, 1)  
# 
# # For a continuous variable, select appropriate percentiles
# # percentiles <- c(.1, .25, .5, .75, .9)
# # grid.size <- length(percentiles)
# # covariate.grid <- quantile(data[,selected.covariate], probs=percentiles)
# 
# # Take median of other covariates 
# medians <- apply(data[, other.covariates, F], 2, median)
# 
# # Construct a dataset
# data.grid <- data.frame(sapply(medians, function(x) rep(x, grid.size)), covariate.grid)
# colnames(data.grid) <- c(other.covariates, selected.covariate)
# 
# # Expand the data
# X.grid <- model.matrix(fmla, data.grid)
# 
# # Point predictions of the CATE and standard errors 
# forest.pred <- predict(forest.tau, newdata = X.grid, estimate.variance=TRUE)
# tau.hat <- forest.pred$predictions
# tau.hat.se <- sqrt(forest.pred$variance.estimates)
# 
# # Plot predictions for each group and 95% confidence intervals around them.
# data.pred <- transform(data.grid, tau.hat=tau.hat, ci.low = tau.hat - 2*tau.hat.se, ci.high = tau.hat + 2*tau.hat.se)
# ggplot(data.pred) +
#   geom_line(aes_string(x=selected.covariate, y="tau.hat", group = 1), color="black") +
#   geom_errorbar(aes_string(x=selected.covariate, ymin="ci.low", ymax="ci.high", width=.2), color="blue") +
#   ylab("") +
#   ggtitle(paste0("Predicted treatment effect varying '", selected.covariate, "' (other variables fixed at median)")) +
#   scale_x_continuous("polviews", breaks=covariate.grid, labels=signif(covariate.grid, 2)) +
#   theme_minimal() +
#   theme(plot.title = element_text(size = 11, face = "bold")) 
# 
# # confidence interval band
# (p2 <- ggplot(mp, aes(wav, wow))+
#     geom_point()+
#     geom_line(data=predframe)+
#     geom_ribbon(data=predframe,aes(ymin=lwr,ymax=upr),alpha=0.3))