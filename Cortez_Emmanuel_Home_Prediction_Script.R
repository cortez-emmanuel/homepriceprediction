# Emmanuel Cortez
# Dr. Garcia
# ADTA 5410
# 14 February 2021

##### Home Price Prediction - Extra Credit #####

King <- read.csv(file.choose('King County Homes (train).csv'))
KingTest <- read.csv(file.choose('King County Homes (test).csv'))

library(leaps)
library(caret)
library(ggplot2)
library(GGally)
library(ggfortify)
library(olsrr)
library(car)
library(Hmisc)
library(psych)
library(corrplot)
library(dplyr)
library(ISLR)
library(MASS)
library(bestNormalize)
library(bestglm)


## Dataset

### King County, Washington Homes

options(scipen=999) # removing scientific notation

str(King)
summary(King)
head(King)

str(KingTest)
summary(KingTest)
head(KingTest)

# Data Type Transformations

# Transformed to Ordinal/Ordered Factors

## Training - Ordinal/Ordered Factors
King$view <- ordered(King$view)
King$condition <- ordered(King$condition)
King$grade <- ordered(King$grade)

## Test - Ordinal/Ordered Factors
KingTest$view <- ordered(KingTest$view)
KingTest$condition <- ordered(KingTest$condition)
KingTest$grade <- ordered(KingTest$grade)

# Understanding the grade variable
typeof(King$grade) # integer
class(King$grade) # ordered factor

# Transformed to Nominal/Factors

## Train - Nominal/Factors
King$zipcode <- as.factor(King$zipcode)
King$waterfront <- as.factor(King$waterfront)
King$renovated <- as.factor(King$renovated)

## Test - Nominal/Factors
KingTest$zipcode <- as.factor(KingTest$zipcode)
KingTest$waterfront <- as.factor(KingTest$waterfront)
KingTest$renovated <- as.factor(KingTest$renovated)

# Confirming updates have been made
str(King)
str(KingTest)

# Deleting Outlier: Observation 11884 with 33 bedrooms

plot(King$price~King$bedrooms)

identify(King$bedrooms, King$price, labels = King$ID)

plot(King$price~King$bedrooms)

summary(King[11884,])

King <- as.data.frame(King[-c(11884),])

plot(King$price~King$bedrooms)

summary(King)

# Listing out fields order for reference

# 1 - ID 2 - price 3 - bedrooms 4 - bathrooms 5- sqft_living 6 - sqft_lot 7 - floors 8 - waterfront 9 - view 10 - condition 11 - grade 12 - sqft_above 13 - sqft_basement 14 - yr_built 15 - yr_renovated 16 - renovated 17 - zipcode 18 - lat 19 - long 20 - sqft_living15 21 - sqft_lot15


# Bar Charts and Boxplots for categorical/ordinal predictors

## Bar Charts

### Frequency of View Rating
view_bar <- ggplot(King, aes(x=view))+
  geom_bar(stat="count", width=0.7,fill="tomato1")+
  geom_text(stat="count",aes(label=after_stat(count)), vjust=-0.25)+
  theme_minimal()+
  ggtitle("Frequency of View Rating") +
  xlab("View") + ylab("Count")


### Frequency of Condition Index
condition_bar <- ggplot(King, aes(x=condition))+
  geom_bar(stat="count",width=0.7,fill="navajowhite4")+
  geom_text(stat="count",aes(label=after_stat(count)),vjust=-0.25)+
  theme_minimal()+
  ggtitle("Frequency of Condition Index")+
  xlab("Condition") + ylab("Count")

### Frequency of Waterfront Status
waterfront_bar <- ggplot(King,aes(x=waterfront))+
  geom_bar(stat="count",width=0.7,fill="cyan3")+
    geom_text(stat="count",aes(label=after_stat(count)),vjust=-0.25)+
  theme_minimal()+
  ggtitle("Frequency of Waterfront Status \n 0 - No | 1 - Yes")+
  xlab("Waterfront") + ylab("Count")

### Frequency of Renovated Status
renovated_bar <- ggplot(King,aes(x=renovated))+
  geom_bar(stat="count",width=0.7,fill="mediumpurple2")+
  geom_text(stat="count",aes(label=after_stat(count)),vjust=-0.25)+
  theme_minimal()+
  ggtitle("Frequency of Renovated Status \n 0 - No | 1 - Yes")+
  xlab("Renovated")+ylab("Count")

view_bar
condition_bar
waterfront_bar
renovated_bar

## Boxplots

boxplot(price, data=King)

par(mfrow=c(2,2))
boxplot(price~view,data=King,main="Price by View")
boxplot(price~condition,data=King,main="Price by Condition")
boxplot(price~waterfront,data=King,main="Price by Waterfront")
boxplot(price~renovated,data=King,main="Price by Renovated")

par(mfrow=c(1,1))
table(King$grade)
ggplot(King, aes(grade)) +
  geom_bar(fill = "#0073C2FF")

# Checking for NA values
# Sidenote: It is good to find that this data set does not have NA values

## Training set - NA
apply(is.na(King),2,sum)
colSums(is.na(King))# sum of NA values which there are none
apply(is.null(King),2,sum)

## Test set - NA
apply(is.na(KingTest),2,sum)
colSums(is.na(KingTest))
apply(is.null(KingTest),2,sum)


#Indexing Continuous/Discrete variables that aren't ordinal and don't have a geo-related value to create correlation matrix
# [2:7,12,13,20,21]

hist(King$price) # this is expected with how most houses are priced in a given DMA or county

# Correlation Matrix

matrix_cols <- c(2:7,12:14,18:21)

ggpairs(King[,matrix_cols])



## We see instances of potential multicollinearity among sets of variables. The more noteworthy ones include: `sqft_living` x `bathrooms` (0.759), `sqft_above` x `sqft_living` (0.876), `sqft_living15` x `sqft_living` (0.754)


# Base Model

King.lm <- lm(price~.-ID-zipcode-lat-long,data=King)

vif(King.lm) # received error that read, "there are aliased coefficients in the model"

alias(King.lm) # main culprits are sqft_living (+1) and sqft_above (-1); the alias function output seems to show that the nonzero values for sqft_living and sqft_above are linearly dependent on sqft_basement; this provides further evidence that sqft_basement contains perfect collinearity

# we will remove sqft_basement from the base model and keep sqft_above and sqft_living for now

King.lm <- lm(price~.-ID-zipcode-lat-long-sqft_basement,data=King)# Base MLR model

vif(King.lm) # VIFs for yr_renovated and renovated are large

King.lm <- lm(price~.-ID-zipcode-sqft_basement-yr_renovated-renovated,data=King)

summary(King.lm)
describe(King) 

# We will ensure we are applying transformations for variables that have strongly skewed distributions for prediction accuracy purposes but at the expense of simplicity. Log transformations will be applied to the following predictors: `sqft_living`, `sqft_lot`, `sqft_living15` and `sqft_lot15`


King.lm <- lm(price ~ bedrooms + bathrooms + log(sqft_living) + log(sqft_lot) + floors + waterfront + view + condition + grade + yr_built + lat + long + log(sqft_living15) + log(sqft_lot15),data=King)

KingII.lm <- lm(log(price) ~ bedrooms + bathrooms + log(sqft_living) + log(sqft_lot) + floors + waterfront + view + condition + grade + yr_built + log(sqft_living15) + log(sqft_lot15),data=King)

#It goes without saying but is important to note that ID and zipcode provide no value as regressors for price. In the case of predicting home price, location might matter but how it's represented in the data can render it an ineffective predictor. 
#yr_renovated and renovated will be excluded given the disproportionate class frequencies. We will rely on our stepwise regression model to trim down the number of terms and use this reduced model within a best subsets selection

# After observing a funnel shaped residual plot, we rid of heteroskedasticity in the model by log transforming the response `price`, otherwise, we can't realistically use predictions

# Diagnostic plots

# Q-Q plots: to view actual residuals divided by their theoretical quantiles if normality were true

par(mfrow=c(1,1))

plot(King.lm, which=1, col=c("royalblue2"))
plot(KingII.lm, which=1, col=c("royalblue2"))

King.resid <- resid(King.lm)
plot(King$bedrooms,King.resid,
     ylab="Residuals",xlab="bedrooms",main="Error of Bedrooms")
     
str(King.lm)

par(mfrow=c(1,2))# 1X2 grid for visuals
autoplot(King.lm)
autoplot(KingII.lm)

summary(King[4323,]) # high leverage moderate to high std error
summary(King[859,]) # high leverage low std error

par(mfrow=c(1,1))

qqnorm(King.lm, pch=1)
qqpline(King.lm,col="red",lwd=2)

?qqnorm()

?autoplot()


par(mfrow=c(1,1))

qqPlot(King.lm,xlab="Normal Quantiles",ylab="Standardized Residuals", main="Normal Q-Q Plot")

?qqPlot()

ols_plot_cooksd_bar(King.lm) # Cook's Distance
ols_plot_dfbetas(King.lm) # dfbetas 
ols_plot_dffits(King.lm) # dffits
ols_plot_resid_stud(King.lm)# studentized residuals


#autoplot(King.lm,which =1:6,label.size=3,data=King,
#        colour='waterfront')

#autoplot(King.lm,which =1:6,label.size=3,data=King,
#        colour='view')

#autoplot(King.lm,which =1:6,label.size=3,data=King,
#        colour='condition')

#autoplot(King.lm,which =1:6,label.size=3,data=King,
#         colour=) grade - let's use a bar chart 

autoplot(King.lm,which =1:6,label.size=3,data=King,
        colour='renovated')

avPlots(King.lm[,matrix_cols])

vif(King.lm[,matrix_cols])

# Trouble Shooting
??framFALSE

# NEXT I will evaluate the base model, stepwise model, and best subsets model's error/predictive performance

# Example formulas can include:
## base model: King.lm <- lm(price ~ . -ID-zipcode-)
## stepwise model: King.step <- step(King.lm)
## best sub model: King.best <- 

# Stepwise 


#King.lm <- lm(price~.-ID-zipcode-sqft_above-sqft_living-sqft_basement,data=King)

## 1b Stepwise Model

King.step <- step(King.lm)
KingII.step <- step(KingII.lm)

step.predII <- predict(KingII.lm,newdata=KingTest)

summary(exp(step.predII))

(err <- exp(step.predII) - King$price)

err2 <- err^2

(RMSE <- sqrt(mean(err^2)))
sd(King$price)

summary(King$residuals)

# Per our stepwise function, it appears that the best model has 1 less term than the base model based on AIC, totaling to 13 variables

King.reducedlm <- lm(price ~ floors + long + bedrooms + log(sqft_lot15) + log(sqft_living15) + condition + bathrooms + log(sqft_living) + waterfront + view + yr_built + lat + grade, data=King)

# Forward Selection - R Base

King.null <- lm(price~1,data=King)# Intercept only model

King.forward <- step(King.null, .~. +bedrooms+bathrooms+log(sqft_living)+log(sqft_lot)+floors+waterfront+view+condition+grade+yr_built+lat+long+log(sqft_living15)+log(sqft_lot15),direction="forward",data=King)

summary(King.forward)

King.forward$anova

# FORWARD ON FULL MODEL

regfit_full <- regsubsets(price~.-sqft_basement, data = King, nvmax = 20)
plot(regfit_full,scale="r2")
plot(regfit_full,scale="adjr2")

plot(King.fwdleaps,scale="r2")
plot(King.fwdleaps,scale="adjr2")

# Forward Selection - Leaps

Kingfwd.leaps 
KingIIfwd.leaps <- regsubsets(log(price) ~ bedrooms + bathrooms + log(sqft_living) + log(sqft_lot) + floors + waterfront + view + condition + grade + yr_built + log(sqft_living15) + log(sqft_lot15), data=King, nvmax=12, method = "forward")

freg.summary <- summary(Kingfwd.leaps)

freg.summaryII <- summary(KingIIfwd.leaps)

which.max(freg.summary$rsq)
which.max(freg.summary$adjr2)
which.min(freg.summary$rss)
which.min(freg.summary$bic)
which.min(freg.summary$cp)

which.max(reg.summaryII$rsq)
which.max(reg.summaryII$adjr2)
which.min(reg.summaryII$rss)
which.min(reg.summaryII$bic)
which.min(reg.summaryII$cp)

Kingbwd.leaps <- regsubsets(price ~ bedrooms + bathrooms + log(sqft_living) + log(sqft_lot) + floors + waterfront + view + condition + grade + yr_built + lat + long + log(sqft_living15) + log(sqft_lot15), data=King, nvmax=14, method = "backward")

breg.summary <- summary(Kingbwd.leaps)

which.max(reg.summary2$rsq)
which.max(reg.summary2$adjr2)
which.min(reg.summary2$rss)
which.min(reg.summary2$bic)
which.min(reg.summary2$cp)

plot(Kingfwd.leaps, scale = "Cp")

val.errors=rep(NA,14)

x.test <- model.matrix(King.lm, data=KingTest)

?model.matrix()

for(i in 1:14){
  coefi <- coef(Kingfwd.leaps,id=i)
  pred <- x.test[,names(coefi)]%*%coefi
  val.errors[i]<- mean((King$price[KingTest]-pred)^2)
}

### Best Subsets Selection - leaps

King.bestsub <- regsubsets(price ~ floors + long + bedrooms + log(sqft_lot15) + log(sqft_living15) + condition + bathrooms + log(sqft_living) + waterfront + view + yr_built + lat + grade, data=King, nvmax=13)

KingII.bestleaps <- regsubsets()

bestsub.summary <- summary(King.bestsub)

which.max(bestsub.summary$adjr2)
which.min(bestsub.summary$rss)
which.min(bestsub.summary$bic)
which.min(bestsub.summary$cp)

par(mfrow=c(2,2))
plot(bestsub.summary$rss,xlab="Number of Terms",ylab="RSS",type="b")
plot(bestsub.summary$adjr2,xlab="Number of Terms",ylab="Adjusted R-square",type="b")
plot(bestsub.summary$cp,xlab="Number of Terms",ylab="Mallow's Cp",type="b")
plot(bestsub.summary$bic,xlab="Number of Terms",ylab="BIC",type="b")

plot(King.bestsub,scale="adjr2")
plot(King.bestsub,scale="Cp")
plot(King.bestsub,scale="bic")

par(mfrow=c(1,1))

# Alternate Forward Stepwise Reduction

########****** 2/21/21 - Try doing the bestglm method with the base model features with AIC; see if there's a way to log indices or if there's an alternate to indexing 

head(King)
y = King[,2]
X = King[,3:14,18:21]
Xy <- cbind(X,y)
King.bestglm <- bestglm(Xy,IC="AIC")
King.bestglm$BestModels

summary(King.bestglm$BestModel)

## 1c Best Subsets Regression # Pg 259-262
King.best <- regsubsets(price ~ grade + lat + view + yr_built + waterfront + bathrooms + condition + bedrooms + long + renovated + floors + sqft_lot,data=King, nvmax = 12)


bestsub_sum <- summary(King.best)

coef(bestsub_sum,10)

names(bestsub_sum)

bestsub_sum$rsq
bestsub_sum$rss
bestsub_sum$adjr2
bestsub_sum$cp
bestsub_sum$bic

par(mfrow=c(2,2))
plot(bestsub_sum$rss,xlab="Number of Terms",ylab="RSS",type="b")
plot(bestsub_sum$adjr2,xlab="Number of Terms",ylab="Adjusted R-square",type="b")
plot(bestsub_sum$cp,xlab="Number of Terms",ylab="Mallow's Cp",type="b")
plot(bestsub_sum$bic,xlab="Number of Terms",ylab="BIC",type="b")

which.max(bestsub_sum$adjr2)
which.min(bestsub_sum$cp)
which.min(bestsub_sum$bic)

plot(King.best,scale="adjr2")# Cp, BIC, 
plot(King.best,scale="r2")
plot(King.best,scale="Cp")
plot(King.best,scale="bic")

## Alternate Best Subset Selection
set.seed(313)

?trainControl()

train.control <- trainControl(method="cv",number=10)
#step.model <- train(price ~ floors + long + bedrooms + log(sqft_lot15) + log(sqft_living15) + condition + bathrooms + log(sqft_living) + waterfront + view + yr_built + lat + grade, data=King,
                    method="leapSeq",
                    tuneGrid=data.frame(nvmax=1:12),
                    trControl=train.control)
step.model$results

step.model$bestTune

# 10 fold Cross Validation

set.seed(313)

tenfold.cv <- trainControl(method = "cv", number = 10)

KingII.cvfit <- train(log(price) ~ bedrooms + bathrooms + log(sqft_living) + log(sqft_lot) + floors + waterfront + view + condition + grade + yr_built + lat + long + log(sqft_living15) + log(sqft_lot15),
                      data=King,
                      method= "lm",
                      trControl= tenfold.cv,
                      metric = "RMSE")

summary(KingII.cvfit)# Examine model
KingII.cvfit# Examine predictions - RMSE Rsq MAE
KingII.cvfit$resample# predictions for each fold
sd(KingII.cvfit$resample$RMSE)# stdev around RMSE
varImp(KingII.cvfit)
plot(varImp(KingII.cvfit), main="Variable Importance")

# RMSE 

summary(KingTest)
str(KingTest)

#my_pred <- predict(twelve.lm, newdata=KingTest)


# Updated Data Types
#str(King)

#my_pred <- predict(twelve.lm, newdata=KingTest)

#summary(my_pred)
#submission <- data.frame(ID=KingTest$ID,ypred=my_pred)
#write.csv(submission,file="DeppasPredictions.csv")





