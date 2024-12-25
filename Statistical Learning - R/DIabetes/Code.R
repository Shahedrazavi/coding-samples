library(data.table)
library(ggplot2)
library(randomForest)
library(leaps)


#Part_1

d_3c = fread('F:/University/01-02/Term6/Statistical Learning (Regression)/Homework/PS2/Data/diabetes_012_health_indicators_BRFSS2015.csv')
d_equal = fread('F:/University/01-02/Term6/Statistical Learning (Regression)/Homework/PS2/Data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
d_unequal = fread('F:/University/01-02/Term6/Statistical Learning (Regression)/Homework/PS2/Data/diabetes_binary_health_indicators_BRFSS2015.csv')

d_3c$Diabetes_012 = as.factor(d_3c$Diabetes_012)
d_equal$Diabetes_binary = as.factor(d_equal$Diabetes_binary)
d_unequal$Diabetes_binary = as.factor(d_unequal$Diabetes_binary)

ggplot(d_3c, aes(BMI, fill = Diabetes_012))+
  geom_density(alpha = .75)

ds_chol = d_3c[, .(n = .N), .(Diabetes_012, HighChol)]
ds_chol[, n_total := sum(n), .(HighChol)]
ds_chol[, n_percent := n / n_total]

ds_phys = d_3c[, .(n = .N), .(Diabetes_012, PhysActivity)]
ds_phys[, n_total := sum(n), .(PhysActivity)]
ds_phys[, n_percent := n / n_total]

ds_stroke = d_3c[, .(n = .N), .(Diabetes_012, Stroke)]
ds_stroke[, n_total := sum(n), .(Stroke)]
ds_stroke[, n_percent := n / n_total]

ggplot(ds_chol, 
       aes(as.factor(HighChol), n_percent, fill = Diabetes_012))+
  geom_bar(stat = 'identity', )

ggplot(ds_phys, 
       aes(as.factor(PhysActivity), n_percent, fill = Diabetes_012))+
  geom_bar(stat = 'identity', )

ggplot(ds_stroke, 
       aes(as.factor(Stroke), n_percent, fill = Diabetes_012))+
  geom_bar(stat = 'identity', )


#Part_2

nrow(d_equal$Diabetes_binary)
data <- d_equal[,-1]
nrow(data)
names(data) <- NULL
nrow(data)
dim(d_equal)
length(d_equal$Diabetes_binary)
length(d_equal$Diabetes_binary)

mtry <- tuneRF(d_equal[,-1],d_equal$Diabetes_binary, ntreeTry=500,
               stepFactor=2,improve=0.05, trace=TRUE, plot=TRUE)
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)

set.seed(71)
rf <-randomForest(Diabetes_binary~.,data=d_equal, ntree=500, mtry=best.m, importance=TRUE)
print(rf)
importance(rf)
varImpPlot(rf)


#Part_3

regfit=regsubsets (Diabetes_binary~.,data=d_equal ,nvmax =21)
reg.sum = summary(regfit)
plot(reg.sum$cp ,xlab =" Number of Variables ",ylab="Cp",
     type='l')
min_cp_index = which.min(reg.sum$cp)
points (min_cp_index, reg.sum$cp [min_cp_index], col ="red",cex =2, pch =20)
predictors = names(coef(regfit, id = min_cp_index))

#Part_4

print(predictors)

idx <- sample(seq(1, 2), size = nrow(d_equal), replace = TRUE, prob = c(.7, .3))
d_train <- d_equal[idx == 1,]
d_test<- d_equal[idx == 2,]

rf = randomForest(as.factor(Diabetes_binary) ~ HighBP+HighChol+CholCheck
                  +BMI+Stroke+HeartDiseaseorAttack+PhysActivity
                  +HvyAlcoholConsump+AnyHealthcare+GenHlth
                  +MentHlth+PhysHlth+DiffWalk+Sex
                  +Age+Education+Income, data = d_train,importance = TRUE , proximity = FALSE)


d_test$predict = predict(rf,d_test)
confusionMatrix(
  factor(d_test$predict, levels = 0:1),
  factor(d_test$Diabetes_binary, levels = 0:1))

