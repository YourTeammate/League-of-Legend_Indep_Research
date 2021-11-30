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



setwd("/home/s06grp/s06_shared/Clarity_Models/C_Model/data")





dev_data<-read_sas("train_t1.sas7bdat")

hol_data<-read_sas("hold_t1.sas7bdat")

oot_data<-read_sas("oot_t1.sas7bdat")

colnames(dev_data)



names(dev_data)<-tolower(names(dev_data))

names(hol_data)<-tolower(names(hol_data))

names(oot_data)<-tolower(names(oot_data))







varlist<-read.csv("/home/s06grp/s06_shared/Clarity_Models/C_Model/R_Code/cmodel_varlist_t1_iter1.csv")

lower_list<-tolower(varlist$varlist)



##REMOVING VARIABLES NOT REQUIRED FOR PREDICTION



df <- subset(dev_data, select = lower_list)

bad<-dev_data$bad_final

weight<-dev_data$weight

#df1 <- subset(df, select = -c(acq_perm_hub_id,appl_vintage,hit_flag,clarity_hit_flag,weight,bad_final,ACCT_APPRV_IND,POPN,fico_band,fico_tag,INET_SCORE_MODEL_CD,INET_SCORE_MODEL_RSN_CD1,INET_SCORE_MODEL_RSN_CD2,INET_SCORE_MODEL_RSN_CD3,INET_SCORE_VAL))

df1 <- subset(df, select = -c(weight,bad_final))



df_oot <- subset(oot_data, select = lower_list)

bad_oot<-oot_data$bad_final

weight_oot<-oot_data$weight

df_oot1 <- subset(df_oot, select = -c(weight,bad_final))





########################## PREPARING HOLD ######################################





df_hold<-subset(hol_data, select = lower_list)

bad_hold<-hol_data$bad_final

weight_hold<-hol_data$weight

df_hold1<-subset(df_hold, select = -c(weight,bad_final))





ddf <- xgb.DMatrix(data=as.matrix(df1),label=bad, missing = NaN,weight=weight)

ddf_hold<-xgb.DMatrix(data=as.matrix(df_hold1),label=bad_hold, missing = NaN,weight=weight_hold)

ddf_oot <- xgb.DMatrix(data=as.matrix(df_oot1),label=bad_oot, missing = NaN,weight=weight_oot)





watchlist <- list(validation=ddf_hold, train=ddf)



monotone_constraint<-varlist$corr



cor<-colnames(ddf)







variable_order<-data.frame(Variable=cor)

variable_order

varlist$Variable<-tolower(varlist$varlist)

varlist



monotone_merge<-merge(variable_order,varlist, by="Variable",sort=FALSE)

monotone_merge







monotone_constraint<-monotone_merge$corr

monotone_constraint

sum(is.na(monotone_constraint))





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





################################################ xgb_fun #################################################



#monotone_constraints=monotone_constraint,





xgb_fun<- function(max.depth, min_child_weight,nrounds,gamma,eta) {
  
  
  
  
  
  param=list ("eta"=eta,
              
              "max.depth"=max.depth,
              
              "gamma"=gamma,
              
              "subsample"=0.7,
              
              "objective"="binary:logistic",
              
              "min_child_weight"= min_child_weight,
              
              tree_method = "hist")
  
  
  
  set.seed(10)
  
  xgb <- xgb.train(monotone_constraints=monotone_constraint,params=param,data=ddf,nrounds=nrounds,verbose=2,eval_metric = "auc", watchlist, maximize=TRUE,nthread=20,early_stopping_rounds=10)
  
  sum<-xgb.importance(feature_names=colnames(as.matrix(df1)),model=xgb)
  
  #monotone_constraints=monotone_constraint,
  
  
  
  ##CALLING KS MACRO
  
  
  
  ##dev prediction
  
  
  
  pred<-predict(xgb,ddf)
  
  score<-round((pred)*1000)
  
  dev<-subset(dev_data,select=c(acq_perm_hub_id,bad_final,weight))
  
  dev1<- cbind(dev,pred,score)
  
  dev_auc<-performance(prediction(pred,dev_data$bad_final),"auc")@y.values[[1]]
  
  ks_dev<-KSMacro(dev1,10,"bad_final","score","weight","acq_perm_hub_id")
  
  
  
  #####hold prediction
  
  
  
  pred<-predict(xgb,ddf_hold)
  
  
  
  score<-round((pred)*1000)
  
  hol<-subset(hol_data,select=c(acq_perm_hub_id,bad_final,weight))
  
  hold_pred<-cbind(hol,pred,score)
  
  
  
  
  
  hold_auc<-performance(prediction(pred,hol_data$bad_final),"auc")@y.values[[1]]
  
  
  
  ks_hold<-KSMacro(hold_pred,10,"bad_final","score","weight","acq_perm_hub_id")
  
  
  
  #####oot prediction
  
  
  
  pred<-predict(xgb,ddf_oot)
  
  
  
  score<-round((pred)*1000)
  
  oot1<-subset(oot_data,select=c(acq_perm_hub_id,bad_final,weight))
  
  oot1_pred<-cbind(oot1,pred,score)
  
  
  
  
  
  oot_auc<-performance(prediction(pred,oot_data$bad_final),"auc")@y.values[[1]]
  
  
  
  ks_oot<-KSMacro(oot1_pred,10,"bad_final","score","weight","acq_perm_hub_id")
  
  
  
  
  
  return(list(Score = ks_hold,
              
              Pred = list(ks_dev,ks_hold,ks_oot,nrow(sum),dev_auc,hold_auc,oot_auc)))
  
  
  
}









start_time<-Sys.time()

set.seed(10)

OPT_Res <- BayesianOptimization(xgb_fun,bounds = list(max.depth = c(2L, 5L),
                                                      
                                                      min_child_weight = c(300L, 3000L),
                                                      
                                                      nrounds=c(50L,250L),
                                                      
                                                      gamma=c(8L,50L),
                                                      
                                                      eta=c(0.05,0.4)),
                                
                                init_grid_dt = NULL, init_points = 30, n_iter =30,
                                
                                acq = "ucb", kappa = 2, eps = 0.0,
                                
                                verbose = TRUE)

end_time<-Sys.time()



end_time-start_time









#OPT_Res

#hist2<-OPT_Res$History[,-5]

hist<-OPT_Res$History



pred_tran<-as.data.frame(t(OPT_Res$Pred))



pred_tran[1]<-unlist(list(pred_tran[1]))

pred_tran[2]<-unlist(list(pred_tran[2]))

pred_tran[3]<-unlist(list(pred_tran[3]))

pred_tran[4]<-unlist(list(pred_tran[4]))

pred_tran[5]<-unlist(list(pred_tran[5]))

pred_tran[6]<-unlist(list(pred_tran[6]))

pred_tran[7]<-unlist(list(pred_tran[7]))



names(pred_tran)<-c("ks_dev","ks_hold","ks_oot","vars","dev_auc","hold_auc","oot_auc")

#hist_2<-cbind(hist2,pred_tran)

hist_all<-cbind(hist,pred_tran)



write.csv(hist_all,"/home/s06grp/s06_shared/Clarity_Models/C_Model/results/kshold_imputed1.csv",row.names = F)





################################# Detail of best model #################################



# use round 46;

#Iter1 Best: eta=0.361305357,max.depth=5,gamma=10,subsample=0.7,min_child_weight= 319,nrounds=148

param=list (eta=0.361305357,
            
            max.depth=5,
            
            gamma=10,
            
            subsample=0.7,
            
            objective="binary:logistic",
            
            min_child_weight= 319,
            
            tree_method ='hist'
            
)



set.seed(10)



xgb <- xgb.train(params=param,data=ddf,nrounds=148,verbose=2,eval_metric = "auc",
                 
                 monotone_constraints=monotone_constraint, watchlist=watchlist, maximize=TRUE,nthread=10,early_stopping_rounds=10)



sum_ri<-xgb.importance(feature_names=colnames(as.matrix(df1)),model=xgb)





write.table(sum_ri,"/home/s06grp/s06_shared/Clarity_Models/C_Model/results/best_1.csv",sep = ",", row.names = F)



pred<-predict(xgb,ddf)