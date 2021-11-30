install.packages("ROCR")

install.packages("rBayesianOptimization")

install.packages("xgboost")

install.packages("haven")

install.packages("data.table")

install.packages("ggplot2")

install.packages("dplyr")



library(haven)

library(xgboost)

library(rBayesianOptimization)

library(ROCR)

library(data.table)

library(ggplot2)

library(lattice)

library(caret)


fp_train = file.path('E:', 'UC San Diego', 'Research', 'LOL', 'data_transform', 'training_data.csv')

fp_test = file.path('E:', 'UC San Diego', 'Research', 'LOL', 'data_transform', 'testing_data.csv')

# oot fp to be added later



dev_data = read.csv(fp_train)
dev_data$weight = rep(1, nrow(dev_data))

hol_data = read.csv(fp_test)
hol_data$weight = rep(1, nrow(hol_data))

# oot data to be added later

### varlist csv
fp_varlist = file.path('E:', 'UC San Diego', 'Research', 'LOL', 'varlist.csv')
varlist = read.csv(fp_varlist)

lower_list = tolower(varlist$varlist)

######################## data pre-processing ############################


################## editing: combine players' winrate together ##################







################################################################################




####################### one hot encoding block (not used for now) #######################
# to_one_hot = c("blue_top_player", "blue_jng_player", "blue_mid_player", "blue_bot_player", "blue_sup_player", 
#                "red_top_player", "red_jng_player", "red_mid_player", "red_bot_player", "red_sup_player")
# to_one_hot

# # get the 25 variables that need to be processed, 10 categorical (players), 15 numerical

# df_raw = subset(dev_data, select = var_list)

# df_raw_hd = subset(hol_data, select = var_list)

# # one hot encoding
# dmy = dummyVars(" ~ .", data = df_raw)

# df = data.frame(predict(dmy, newdata = df_raw))

# df_hold = data.frame(predict(dmy, newdata = df_raw_hd))


# tester = intersect(df_raw$blue_top_player, df_raw_hd$blue_top_player)
# t = c(df_raw_hd$blue_top_player)
# t[! t %in% c(tester)]

# df_hold$blue_top_pl
##########################################################################################

### training data ###

df = subset(dev_data, select = lower_list)

res = dev_data$result

weight = dev_data$weight

df1 = subset(df, select = -c(weight, result))

### testing data ###

df_hold = subset(hol_data, select = lower_list)

res_hold = hol_data$result

weight_hold = hol_data$weight

df_hold1 = subset(df_hold, select = -c(weight, result))



### convert to ddf

ddf = xgb.DMatrix(data=as.matrix(df1),label=res, missing = NaN,weight=weight)

ddf_hold = xgb.DMatrix(data=as.matrix(df_hold1),label=res_hold, missing = NaN,weight=weight_hold)


### monotone constrains (might not be needed)

watchlist = list(validation=ddf_hold, train=ddf)

monotone_constraint = varlist$corr

cor = colnames(ddf)


variable_order = data.frame(Variable=cor)

variable_order

varlist$Variable = tolower(varlist$varlist)

varlist



monotone_merge = merge(variable_order,varlist, by="Variable",sort=FALSE)

monotone_merge



monotone_constraint = monotone_merge$corr

monotone_constraint

sum(is.na(monotone_constraint))



##########testing##########

head(df1)

mean(res)

cor_matrix = cor(df1)

cor_matrix_with_result = cor(subset(df, select = -c(weight)))
# fp_cor = file.path('E:', 'UC San Diego', 'Research', 'LOL', 'cor_matrix.csv')

# write.csv(cor_matrix, fp_cor, row.names = T)

###########################







################################################ KSMacro #################################################





KSMacro<-function(data_raw_path,num_quantiles,badvar_name,scorevar_name,weightvar_name,keyvar_name) ###Score has to be numeric. Use (1-pred)*1000 for the score, in case not like this already###
  
{
  
  
  
  ###Create Subset for removing duplicate rows####
  
  
  
  data_raw<-data_raw_path
  
  
  
  keys<-data_raw[,names(data_raw) %in% keyvar_name]
  
  
  
  data_raw$keyvals<-keys
  
  
  
  a<-which(names(data_raw) %in% badvar_name)
  
  
  
  b<-which(names(data_raw) %in% weightvar_name)
  
  
  
  tot<-c(a,b)
  
  
  
  data_subset<-subset(data_raw,!(data_raw[,tot[1]]==0 & data_raw[,tot[2]]<1))
  
  
  
  print("Check 1")
  
  
  
  ###Create weighted_bad variable###
  
  data_subset$wbad<-data_subset[,names(data_subset) %in% badvar_name]*data_subset[,names(data_subset) %in% weightvar_name]
  
  
  
  print("Check 2")
  
  
  
  ####Binning the score#####
  
  
  
  scores_data<-data_subset[,names(data_subset) %in% scorevar_name]
  
  
  
  #quantile1<-quantile(data_SubsetBad$score,prob=seq(0,1,length=num_quantiles+1),type=8)
  
  
  
  data_subset$quantile1<-with(data_subset,cut(scores_data,breaks=unique(quantile(scores_data,probs=seq(0,1,length=num_quantiles+1),type=8)),include.lowest = TRUE))
  
  data_subset$quantilenum<-as.integer(data_subset$quantile1)
  
  
  
  print("Check 3")
  
  
  
  #####Bad distribution######
  
  
  
  cumpct_bad<-vector(mode="numeric",length=length(unique(data_subset$quantilenum)))
  
  cumpct_bad[1]<-sum(data_subset$wbad[which(data_subset$quantilenum==1)])/sum(data_subset$wbad)
  
  
  
  for(i in 2:length(unique(data_subset$quantilenum)))
    
  {
    
    cumpct_bad[i]<-cumpct_bad[i-1]+(sum(data_subset$wbad[which(data_subset$quantilenum==i)])/sum(data_subset$wbad))
    
  }
  
  
  
  print("Check 4")
  
  
  
  ####Good distribution######
  
  
  
  cumpct_good<-vector(mode="numeric",length=length(unique(data_subset$quantilenum)))
  
  good_denom<-(length(data_subset$keyvals)-sum(data_subset$wbad))
  
  cumpct_good[1]<-(length(data_subset$keyvals[which(data_subset$quantilenum==1)])-sum(data_subset$wbad[which(data_subset$quantilenum==1)]))/(good_denom)
  
  
  
  for(i in 2:length(unique(data_subset$quantilenum)))
    
  {
    
    cumpct_good[i]<-cumpct_good[i-1]+(length(data_subset$keyvals[which(data_subset$quantilenum==i)])-sum(data_subset$wbad[which(data_subset$quantilenum==i)]))/(good_denom)
    
  }
  
  
  
  print("Check 5")
  
  
  
  ###KS statistic###
  
  
  
  ks<-max(abs(cumpct_good-cumpct_bad))
  
  return(ks*100)
  
  
  
}


####################### testing block #########################

param=list ("eta"= 0.2,
            
            "max.depth"= 50,
            
            "gamma"= 20,
            
            "subsample"= 0.7,
            
            "objective"="binary:logistic",
            
            "min_child_weight"= 300,
            
            "tree_method" = "hist")

set.seed(10)

xgb <- xgb.train(params=param,data=ddf,nrounds=100,verbose=2,eval_metric = "auc", watchlist, maximize=TRUE,nthread=20,early_stopping_rounds=10)

sum <- xgb.importance(feature_names=colnames(as.matrix(df1)),model=xgb)

#################################################################




################################################ xgb_fun #################################################




xgb_fun <- function(max.depth, min_child_weight,nrounds,gamma,eta) {
  
  param=list ("eta"=eta,
              
              "max.depth"=max.depth,
              
              "gamma"=gamma,
              
              "subsample"=0.7,
              
              "objective"="binary:logistic",
              
              "min_child_weight"= min_child_weight,
              
              "tree_method" = "hist")
  

  set.seed(10)
  
  xgb <- xgb.train(params=param,data=ddf,nrounds=nrounds,verbose=2,eval_metric = "auc", watchlist, maximize=TRUE,nthread=20,early_stopping_rounds=10)
  
  sum <- xgb.importance(feature_names=colnames(as.matrix(df1)),model=xgb)
  
  #monotone_constraints=monotone_constraint,
  
  
  
  ##CALLING KS MACRO
  
  
  
  ##dev prediction
  
  
  
  pred<-predict(xgb,ddf)
  
  score<-round((pred)*1000)
  
  dev<-subset(dev_data,select=c(X, result, weight))
  
  dev1<- cbind(dev,pred,score)
  
  dev_auc<-performance(prediction(pred,dev_data$result),"auc")@y.values[[1]]
  
  ks_dev<-KSMacro(dev1,10,"result","score","weight","X")
  
  
  
  #####hold prediction
  
  
  
  pred<-predict(xgb,ddf_hold)
  
  score<-round((pred)*1000)
  
  hol<-subset(hol_data,select=c(X, result, weight))
  
  hold_pred<-cbind(hol,pred,score)
  
  hold_auc<-performance(prediction(pred,hol_data$result),"auc")@y.values[[1]]
  
  ks_hold<-KSMacro(hold_pred,10,"result","score","weight","X")
  
  
  
  # return(list(Score = ks_hold,
              
  #             Pred = list(ks_dev,ks_hold,ks_oot,nrow(sum),dev_auc,hold_auc,oot_auc)))
  return(list(Score = ks_hold,
              
              Pred = list(ks_dev, ks_hold, nrow(sum), dev_auc, hold_auc)))
  
  
}





######################## Bayesian Optimization ############################

start_time = Sys.time()

set.seed(10)

OPT_Res = BayesianOptimization(xgb_fun,bounds = list(max.depth = c(2L, 7L),
                                                      
                                                      min_child_weight = c(20L, 300L),
                                                      
                                                      nrounds=c(50L,250L),
                                                      
                                                      gamma=c(8L,50L),
                                                      
                                                      eta=c(0.05,0.4)),
                                
                                init_grid_dt = NULL, init_points = 30, n_iter =30,
                                
                                acq = "ucb", kappa = 2, eps = 0.0,
                                
                                verbose = TRUE)



## detailed opt ##

OPT_Res = BayesianOptimization(xgb_fun,bounds = list(max.depth = c(6L, 10L),
                                                     
                                                     min_child_weight = c(250L, 400L),
                                                     
                                                     nrounds=c(20L,100L),
                                                     
                                                     gamma=c(4L,10L),
                                                     
                                                     eta=c(0.35,0.55)),
                               
                               init_grid_dt = NULL, init_points = 30, n_iter =30,
                               
                               acq = "ucb", kappa = 2, eps = 0.0,
                               
                               verbose = TRUE)

end_time = Sys.time()









################ result observation ########################

hist<-OPT_Res$History

pred_tran<-as.data.frame(t(OPT_Res$Pred))



pred_tran[1]<-unlist(list(pred_tran[1]))

pred_tran[2]<-unlist(list(pred_tran[2]))

pred_tran[3]<-unlist(list(pred_tran[3]))

pred_tran[4]<-unlist(list(pred_tran[4]))

pred_tran[5]<-unlist(list(pred_tran[5]))

#pred_tran[6]<-unlist(list(pred_tran[6]))

#pred_tran[7]<-unlist(list(pred_tran[7]))



#names(pred_tran)<-c("ks_dev","ks_hold","ks_oot","vars","dev_auc","hold_auc","oot_auc")
names(pred_tran)<-c("ks_dev","ks_hold","vars","dev_auc","hold_auc")

#hist_2<-cbind(hist2,pred_tran)

hist_all<-cbind(hist,pred_tran)


fp_opt_res = file.path('E:', 'UC San Diego', 'Research', 'LOL', 'kshold_imputed1.csv')

write.csv(hist_all, fp_opt_res, row.names = F)


#################################################



####################### Logistic Regression ############################

logis_model = glm(result ~., family = binomial(link = 'logit'), data = subset(df, select = -c(weight)))

logis_model = glm(result ~., family = binomial(link = 'logit'), data = subset(df, select = -c(weight, blue_jng_player, jng_counter_winrate, top_counter_winrate)))

summary(logis_model)


# get accuracy

fit_test = predict(logis_model, newdata = subset(df_hold, select = -c(weight, result)))

fit_test = ifelse(fit_test > 0.5, 1, 0)

log_accuracy = mean(fit_test == df_hold$result)


# get auc

log_p = predict(logis_model, newdata = subset(df_hold, select = -c(weight, result)), type = 'response')

log_pred = prediction(log_p, df_hold$result)

log_auc = performance(log_pred, measure = 'auc')

log_auc_value = log_auc@y.values[[1]]


# roc

pred <- ROCR::prediction(predictions=log_p, labels=actual_hold)

perf <- performance(pred, measure = "tpr", x.measure = "fpr") 

plot(perf, col=rainbow(10))


########################################################################





######################### get the accuracy #############################

OPT_Res$Best_Par

# use the best parameter

param=list ("eta"= 0.4,
            
            "max.depth"= 6,
            
            "gamma"= 38,
            
            "subsample"= 0.7,
            
            "objective"="binary:logistic",
            
            "min_child_weight"= 274,
            
            "tree_method" = "hist")

set.seed(10)

xgb <- xgb.train(params=param,data=ddf,nrounds=134,verbose=2,eval_metric = "auc", watchlist, maximize=TRUE,nthread=20,early_stopping_rounds=10)

sum <- xgb.importance(feature_names=colnames(as.matrix(df1)),model=xgb)

fp_sum = file.path('E:', 'UC San Diego', 'Research', 'LOL', 'best_model.csv')

write.table(sum, fp_sum, row.names = F)




pred = predict(xgb, ddf)

pred_rounded = ifelse(pred > 0.5, 1, 0)

actual_dev = dev_data$result

accuracy_dev = mean(actual_dev == pred_rounded)


pred_hold = predict(xgb, ddf_hold)

pred_rounded_hold = ifelse(pred_hold > 0.5, 1, 0)

actual_hold = hol_data$result

accuracy_hold = mean(actual_hold == pred_rounded_hold)


########################################################################

# run 46

param=list ("eta"= 0.4,
            
            "max.depth"= 7,
            
            "gamma"= 8,
            
            "subsample"= 0.7,
            
            "objective"="binary:logistic",
            
            "min_child_weight"= 300,
            
            "tree_method" = "hist")

set.seed(10)

xgb <- xgb.train(params=param,data=ddf,nrounds=50,verbose=2,eval_metric = "auc", watchlist, maximize=TRUE,nthread=20,early_stopping_rounds=10)

sum <- xgb.importance(feature_names=colnames(as.matrix(df1)),model=xgb)

fp_sum = file.path('E:', 'UC San Diego', 'Research', 'LOL', 'model_round_46.csv')

write.table(sum, fp_sum, row.names = F)


pred_hold = predict(xgb, ddf_hold)

pred_rounded_hold = ifelse(pred_hold > 0.5, 1, 0)

actual_hold = hol_data$result

accuracy_hold = mean(actual_hold == pred_rounded_hold)


pred <- ROCR::prediction(predictions=pred_hold, labels=actual_hold)

perf <- performance(pred, measure = "tpr", x.measure = "fpr") 

plot(perf, col=rainbow(10))



