#Install Packages from http://r4ds.had.co.nz/
r4dspackages <- c("tidyverse", "nycflights13", "gapminder", "Lahman")
install.packages(r4dspackages)
lapply(r4dspackages, library, character.only=T)

#Install Packages from W13 Tips
w13packages <- c("leaps", "FSelector", "caret", "rpart", "pROC", "mlbench")
install.packages(w13packages)
lapply(w13packages, library, character.only=T)

rm(list = ls())

#Read in the dataset
setwd("C:/Users/Raymond Kirk/Desktop")

# load libraries
rm(list = ls())
library(caret)
library(pROC)

#################################################
# Tidy
#################################################

#Load data
df <- read.csv('ds_training.csv')

#Remove ID (100.0% Unique)
df1 <- df[, -which(names(df) %in% "ID")]

#Check for NA                  
if(sum(is.na(df)) > 0) na.omit(df)

#Remove Near Zero Variance Variables
nzv_cols <- nearZeroVar(df1[, -which(names(df1) %in% "TARGET")])
if(length(nzv_cols[-1]) > 0) df2 <- df1[, -nzv_cols]

#Get Proportions of the outcome variable 
prop.table(table(df2$TARGET))

outcomeName <- 'TARGET'
predictorsNames <- names(df2)[names(df2) != outcomeName]

# save the outcome for the glmnet model - Remove Later
tempOutcome <- df2$TARGET  

#################################################
# Model Relationships and Distributions
#################################################

df2$TARGET <- ifelse(df2$TARGET==1,'ONE','ZERO')

#Split data into training and testing (75/25 Split)
set.seed(2580)
splitIndex <- createDataPartition(df2[,outcomeName], p = .75, list = FALSE, times = 1)
trainDF <- df2[ splitIndex,]
testDF  <- df2[-splitIndex,]

#trainControl handles cross validations performed on training.
objControl <- trainControl(method='cv', number=3, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)

#Train model
objModel <- train(trainDF[,predictorsNames], as.factor(trainDF[,outcomeName]), 
                  method='gbm', 
                  trControl=objControl,  
                  metric = "ROC",
                  preProc = c("center", "scale"))


#List Importance of Variables
summary(objModel)

#################################################
# Evaluate Model
#################################################

#TARGET Prediction on testing data
predictions <- predict(object=objModel, testDF[,predictorsNames], type='raw')
head(predictions)
postResample(pred=predictions, obs=as.factor(testDF[,outcomeName]))

#Proablities and AUC
predictions <- predict(object=objModel, testDF[,predictorsNames], type='prob')
head(predictions)
postResample(pred=predictions[[2]], obs=ifelse(testDF[,outcomeName]=='ONE',1,0))
auc <- roc(ifelse(testDF[,outcomeName]=="ONE",1,0), predictions[[2]])
print(auc$auc)

######################################################

df2Test <- df2

# pick model gbm and find out what type of model it is
getModelInfo()$glmnet$type

# save the outcome for the glmnet model
df2Test$TARGET  <- tempOutcome

# split data into training and testing chunks
set.seed(1234)
splitIndex <- createDataPartition(df2Test[,outcomeName], p = .75, list = FALSE, times = 1)
trainDF <- df2Test[ splitIndex,]
testDF  <- df2Test[-splitIndex,]

# create caret trainControl object to control the number of cross-validations performed
objControl <- trainControl(method='cv', number=3, returnResamp='none')

# run model
objModel <- train(trainDF[,predictorsNames], trainDF[,outcomeName], method='lasso',  metric = "RMSE", trControl=objControl)

# get predictions on your testing data
predictions <- predict(object=objModel, testDF[,predictorsNames])

library(pROC)
auc <- roc(testDF[,"TARGET"], predictions)
print(auc$auc)

postResample(pred=predictions, obs=testDF[,outcomeName])

# find out variable importance
summary(objModel)
plot(varImp(objModel,scale=F))

# find out model details
objModel

# display variable importance on a +/- scale 
vimp <- varImp(objModel, scale=F)
results <- data.frame(row.names(vimp$importance),vimp$importance$Overall)
results$VariableName <- rownames(vimp)
colnames(results) <- c('VariableName','Weight')
results <- results[order(results$Weight),]
results <- results[(results$Weight != 0),]

par(mar=c(5,15,4,2)) # increase y-axis margin. 
xx <- barplot(results$Weight, width = 0.85, 
              main = paste("Variable Importance -",outcomeName), horiz = T, 
              xlab = "< (-) importance >  < neutral >  < importance (+) >", axes = FALSE, 
              col = ifelse((results$Weight > 0), 'blue', 'red')) 
axis(2, at=xx, labels=results$VariableName, tick=FALSE, las=2, line=-0.3, cex.axis=0.6)  
