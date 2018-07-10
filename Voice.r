##Reading the dataset
voicedata <- read.csv("voice.csv")
View(voicedata)
library(caret)
library(kernlab)
library(adabag)
library(nnet)
library(e1071)
library(dplyr)
library(tidyr)
library(tree)
library(rpart.plot)

##check any missing data 
voicedata[!complete.cases(voicedata),]

# create new dataset without missing data 
voicedata1 <- na.omit(voicedata)

##--##--####--##--####--##--####--##--####--##--####--##--####--##--####--##--####--##--##
##Splitting the dataset
voice <- read.csv("voice.csv")
##data partition
index <- createDataPartition(voice$label, p = 0.75, list = FALSE)
test <- voicedata1[-index, ]
train <- voicedata1[index, ]

dim(test)
dim(train)
## y contains the class label which is the response attribute
x <- train[, 1:20]
y <- train[, 21]

##checking the dimensions and summary of the voice dataset
dim(voice)
str(voice)
summary(voice)
table(voice$label)

## Density plot of attributes for male and female
voice %>% na.omit() %>%
  gather(type,value,1:20) %>%
  ggplot(aes(x=value,fill=label))+geom_density(alpha=0.3)+facet_wrap(~type,scales="free")
+theme(axis.text.x = element_text(angle = 90,vjust=1))
+labs(title="Density Plots of Data across Variables")

## it means there is a lot of correlation as most of the attributes overlap for male and female

##creating the first model
set.seed(100)
control <- trainControl(method="cv", number=12)
metric <- "Accuracy"
##Linear Discriminant Analysis
model.lda <- train(label~., data=train, method="lda", metric=metric, trControl=control)
prediction.lda <- predict(model.lda, test)
lda.acc=confusionMatrix(prediction.lda, test$label)$overall[1]

confusionMatrix(prediction.lda, test$label)
##CART
model.rpart <- train(label~., data=train, method="rpart", metric=metric, trControl=control)
prediction.rpart <- predict(model.rpart, test)
rpart.acc=confusionMatrix(prediction.rpart, test$label)$overall[1]
confusionMatrix(prediction.rpart, test$label)

form <- as.formula(label ~ .)
tree.1 <- rpart(form,data=train,control=rpart.control(minsplit=20,cp=0))
# 
plot(tree.1)					# Will make a mess of the plot
text(tree.1)

prp(tree.1)					# Will plot the tree
prp(tree.1,varlen=3)


##KNN
model.knn <- train(label~., data=train, method="knn", metric=metric, trControl=control)
prediction.knn <- predict(model.knn, test)
knn.acc=confusionMatrix(prediction.knn, test$label)$overall[1]
confusionMatrix(prediction.knn, test$label)
##SVM
model.svm <- train(label~., data=train, method="svmRadial", metric=metric, trControl=control)
prediction.svm <- predict(model.svm, test)
svm.acc=confusionMatrix(prediction.svm, test$label)$overall[1]
confusionMatrix(prediction.svm, test$label)
#RF
model.rf <- train(label~., data=train, method="rf", metric=metric, trControl=control)
prediction.rf <- predict(model.rf, test)
rf.acc=confusionMatrix(prediction.rf, test$label)$overall[1]
confusionMatrix(prediction.rf, test$label)
#FDA
model.fda <- train(label~., data=train, method="fda", metric=metric, trControl=control)
prediction.fda <- predict(model.fda, test)
fda.acc=confusionMatrix(prediction.fda, test$label)$overall[1]
confusionMatrix(prediction.fda, test$label)
#NBA
model.naiveBayes <- naiveBayes(label~., data=train, metric=metric, trControl=control)
prediction.NB <- predict(model.naiveBayes, test)
nba.acc=confusionMatrix(prediction.NB, test$label)$overall[1]
confusionMatrix(prediction.NB, test$label)
#C50
model.c50 <- train(label~., data=train, method="C5.0", metric=metric, trControl=control)
prediction.c50 <- predict(model.c50, test)
c50.acc=confusionMatrix(prediction.c50, test$label)$overall[1]
confusionMatrix(prediction.c50, test$label)

# collect resamples
results <- (list(lda=lda.acc,rpart=rpart.acc, NB=nba.acc,C5.0t=c50.acc,KNN=knn.acc,SVMRadial=svm.acc))

# summarize the distributions
View(results)

model_results <- resamples(list(lda=model.lda, rpart=model.rpart,KNN=model.knn,SVM=model.svm,RF=model.rf,FDA=model.fda,C50=model.c50))
summary(model_results)
dotplot(model_results)

##overall accuracy is better for SVM
##SVM has the highest accuracy

library(corrplot)
### visual exploration of the dataset for correlation
predictor_Corr <- cor(voicedata1[,-21])
corrplot(predictor_Corr,method="number")

### pre-process of original dataset to remove correlations
pca_Transform <- preProcess(voicedata1,method=c("scale","center","pca"))
voicedata2 <- predict(pca_Transform,voicedata1)
View(voicedata2)
head(voicedata2)

##trying SVM after removing correlation
##SVM

index <- createDataPartition(voice$label, p = 0.75, list = FALSE)
test <- voicedata1[-index, ]
train <- voicedata1[index, ]

x <- train[, 1:20]
y <- train[, 21]


model.svm <- train(label~., data=train, method="svmRadial", metric=metric, trControl=control)
prediction.svm <- predict(model.svm, test)
svm.acc=confusionMatrix(prediction.svm, test$label)$overall[1]



### split original dataset into training and testing subsets
sample_Index <- createDataPartition(voicedata2$label,p=0.75,list=FALSE)
voice_Train <- voicedata2[sample_Index,]
voice_Test <- voicedata2[-sample_Index,]

### visual explorations
# correlation plot
new_Corr <- cor(voicedata2[,2:11])
corrplot(new_Corr)

##voicedata2
  ggplot(voicedata2,aes(x=PC1,y=PC2))+
  geom_point(aes(color=label))

##voice_Original%>%
  ggplot(voicedata2,aes(x=PC2,y=PC3))+
  geom_point(aes(color=label))

# set formula
model_Formula <- label~PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8+PC9+PC10

### set tuning and cross validation paramters
modelControl <- trainControl(method="cv",number=12)

## model 1: logistic regression
glm_Model <- train(
  model_Formula,
  data=voicedata2,
  method="glm",
  trControl=modelControl
)

## prediction with glm model
voice_Test$glmPrediction <- predict(glm_Model,newdata=voice_Test[,2:11])
## view prediction results
glm_Model   ### accuracy 0.9709 kappa 0.9418
table(voice_Test$label,voice_Test$glmPrediction)
glm.acc=confusionMatrix(voice_Test$glmPrediction,voice_Test$label)$overall[1]
glm.acc

## model 2: random forest
rf_Model <- train(
  model_Formula,
  data=voice_Train,
  method="rf",
  ntrees=1000,
  trControl=modelControl
)
voice_Test$rfPrediction <- predict(rf_Model,newdata=voice_Test[,2:11])
## view model performance
rf_Model  ## accuracy 0.967 kappa 0.934
table(voice_Test$label,voice_Test$rfPrediction)
RF.acc=confusionMatrix(voice_Test$rfPrediction,voice_Test$label)$overall[1]
RF.acc

## model 3: support vector machine
svm_Model <- train(
  model_Formula,
  data=voice_Train,
  method="svmRadial",
  trControl=modelControl
)
## view model performance
svm_Model  ## accuracy 0.974 kappa 0.949
voice_Test$svmPrediction <- predict(svm_Model,newdata=voice_Test[,2:11])
table(voice_Test$label,voice_Test$svmPrediction)
svm.acc=confusionMatrix(voice_Test$svmPrediction,voice_Test$label)$overall[1]
svm.acc
## model 4: gradient boosting machine
gbm_Model <- train(
  model_Formula,
  data=voice_Train,
  method="gbm",
  trControl=modelControl
)
## view model performance
gbm_Model  ## best performance @ accuracy 0.968 kappa 0.935
voice_Test$gbmPrediction <- predict(gbm_Model,newdata=voice_Test[,2:11])
table(voice_Test$label,voice_Test$gbmPrediction)
gbm.acc=confusionMatrix(voice_Test$gbmPrediction,voice_Test$label)$overall[1]
gbm.acc

#model 5:C50
c50_Model <- train(
  model_Formula,
  data=voice_Train,
  method="C5.0",
  trControl=modelControl
)
## view model performance
c50_Model  ## best performance @ accuracy 0.968 kappa 0.935
voice_Test$c50Prediction <- predict(c50_Model,newdata=voice_Test[,2:11])
table(voice_Test$label,voice_Test$c50Prediction)
c50.acc=confusionMatrix(voice_Test$c50Prediction,voice_Test$label)$overall[1]
c50.acc

### compare model performance of 4 models that have been built
model_Comparison <- resamples(
  list(
    LogisticRegression=glm_Model,
    RandomForest=rf_Model,
    SupportVectorMachine=svm_Model,
    GradientBoosting=gbm_Model,
    C5.0trees=c50_Model
  )
)

summary(model_Comparison)
results <- (list(LogisticRegression=glm_Model,
                 RandomForest=rf_Model,
                 SupportVectorMachine=svm_Model,
                 GradientBoosting=gbm_Model,
                 C5.0trees=c50_Model))

View(results)
## visual comparison of model performances
bwplot(model_Comparison,layout=c(2,1))

##-####-####-####-####-####-####-####-####-####-####-####-####-##
##starting from scratch
##Splitting the dataset
voice <- read.csv("voice.csv")
##data partition
index <- createDataPartition(voice$label, p = 0.75, list = FALSE)
test <- voicedata1[-index, ]
train <- voicedata1[index, ]

dim(test)
dim(train)
## y contains the class label which is the response attribute
x <- train[, 1:20]
y <- train[, 21]

##checking the dimensions and summary of the voice dataset
dim(voice)
str(voice)
summary(voice)
table(voice$label)

## Density plot of attributes for male and female
voice %>% na.omit() %>%
  gather(type,value,1:20) %>%
  ggplot(aes(x=value,fill=label))+geom_density(alpha=0.3)+facet_wrap(~type,scales="free")+theme(axis.text.x = element_text(angle = 90,vjust=1))+labs(title="Density Plots of Data across Variables")

## it means there is a lot of correlation as most of the attributes overlap for male and female

##creating the first model
set.seed(100)
control <- trainControl(method="cv", number=12)
metric <- "Accuracy"
#finding out PCA
voice$label=as.factor(voice$label)
#divide train and test
idx=createDataPartition(voice$label,p=0.75,list=FALSE)
train_data=voice[idx,]
test_data=voice[-idx,]
install.packages("Boruta")
library(Boruta)
boruta.train <- Boruta(label~., data = train_data, doTrace = 2)
print(boruta.train)
plot(boruta.train)

##variable importance using random forest
library(randomForest)
index <- createDataPartition(voice$label, p = 0.75, list = FALSE)

test <- voice[-index, ]
train <- voice[index, ]

x <- train[, 1:20]
y <- train[, 21]

table(train$label)/nrow(train)
table(test$label)/nrow(test)
set.seed(3)
model <- randomForest(label~., train, ntree = 120, importance = T)
plot(model)

varImpPlot(model, sort = T, main="Variable Importance - Accuracy", n.var=6, type = 1)
#using SVM only on the prime attributes

# set formula only for prime components
model_Formula <- label~meanfun+IQR+Q25+sd+sp.ent+sfm
svm_Model <- train(
  model_Formula,
  data=train,
  method="svmRadial",
  trControl=modelControl
)
## view model performance
svm_Model## accuracy 0.974 kappa 0.949
voice_Train$svmPrediction<-predict(svm_Model,newdata=train[,1:20])
table(test$label,voice_Test$svmPrediction)
svm.acc=confusionMatrix(voice_Test$svmPrediction,test$label)$overall[1]
svm.acc

  