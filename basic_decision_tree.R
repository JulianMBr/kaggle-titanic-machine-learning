library(e1071)
library(rpart)
library(rattle)
library(rpart.plot)
library(party)
library(Hmisc)
library(RColorBrewer)
library(randomForest)

train <- read.csv("train.csv")
test <- read.csv("test.csv")

#checking the missing data
check.missing<-function(x) return(paste0(round(sum(is.na(x))/length(x),4)*100,'%'))
data.frame(sapply(train,check.missing))
data.frame(sapply(test,check.missing))

# Join together the test and train sets for easier feature engineering
test$Survived <- NA
combi <- rbind(train, test)

# Convert to a string
combi$Name <- as.character(combi$Name)

# Engineered variable: Title
combi$Title <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
combi$Title <- sub(' ', '', combi$Title)

# Combine small title groups
combi$Title[combi$PassengerId == 797] <- 'Mrs' # female doctor
combi$Title[combi$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir', 'Jonkheer')] <- 'Sir'
combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess')] <- 'Lady'

# Convert to a factor
combi$Title <- factor(combi$Title)

# Engineered variable: Family size
combi$FamilySize <- combi$SibSp + combi$Parch + 1

# Engineered variable: Family
combi$Surname <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep="")
combi$FamilyID[combi$FamilySize <= 2] <- 'Small'
# Delete erroneous family IDs
famIDs <- data.frame(table(combi$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] <- 'Small'
# Convert to a factor
combi$FamilyID <- factor(combi$FamilyID)


# Fill in Age NAs
summary(combi$Age)
bystats(combi$Age, combi$Title, fun=function(x)c(Mean=mean(x),Median=median(x)))


# predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
#                         data=combi[!is.na(combi$Age),], method="anova")
# combi$Age[is.na(combi$Age)] <- predict(predicted_age, combi[is.na(combi$Age),])

titles.na.train <- c("Dr", "Master", "Mrs", "Miss", "Mr", "Ms")

imputeMedian <- function(impute.var, filter.var, var.levels) {
  for (v in var.levels) {
    impute.var[ which( filter.var == v)] <- impute(impute.var[which( filter.var == v)])
  }
  return (impute.var)
}

combi$Age <- imputeMedian(combi$Age, combi$Title, titles.na.train)


# Fill in Embarked blanks
summary(combi$Embarked)
which(combi$Embarked == '')
combi$Embarked[c(62,830)] = "S"
combi$Embarked <- factor(combi$Embarked)
# Fill in Fare NAs
summary(combi$Fare)
which(is.na(combi$Fare))
combi$Fare[1044] <- median(combi$Fare, na.rm=TRUE)

# New factor for Random Forests, only allowed <32 levels, so reduce number
combi$FamilyID2 <- combi$FamilyID
# Convert back to string
combi$FamilyID2 <- as.character(combi$FamilyID2)
combi$FamilyID2[combi$FamilySize <= 3] <- 'Small'
# And convert back to factor
combi$FamilyID2 <- factor(combi$FamilyID2)

## adding mother / child column
combi$Mother<-0
combi$Mother[combi$Sex=='female' & combi$Parch>0 & combi$Age>18 & combi$Title!='Miss']<-1
#Adding Child
combi$Child<-0
combi$Child[combi$Parch>0 & combi$Age<=18]<-1

# #Exact Deck from Cabin number
combi$Cabin <- as.character(combi$Cabin)
combi$Deck<-sapply(combi$Cabin, function(x) strsplit(x,NULL)[[1]][1])
deck.fit<-rpart(Deck~Pclass+Fare,data=combi[!is.na(combi$Deck),])
combi$Deck[is.na(combi$Deck)]<-as.character(predict(deck.fit,combi[is.na(combi$Deck),],type='class'))
combi$Deck[is.na(combi$Deck)]<-'UNK'


# #Excat Position from Cabin number
combi$CabinNum<-sapply(combi$Cabin,function(x) strsplit(x,'[A-Z]')[[1]][2])
combi$num<-as.numeric(combi$CabinNum)
num<-combi$num[!is.na(combi$num)]
Pos<-kmeans(num,3)
combi$CabinPos[!is.na(combi$num)]<-Pos$cluster
combi$CabinPos<-factor(combi$CabinPos)
levels(combi$CabinPos)<-c('Front','End','Middle')
combi$num<-NULL

## remove NAs
combi$CabinPos = factor(combi$CabinPos, levels=c(levels(combi$CabinPos), 0))
combi$CabinPos[is.na(combi$CabinPos)] = 0
combi$CabinNum = factor(combi$CabinNum, levels=c(levels(combi$CabinNum), 0))
combi$CabinNum[is.na(combi$CabinNum)] = 0

## re-factor
combi$Cabin <- as.factor(combi$Cabin)
combi$CabinNum <- as.factor(combi$CabinNum)
combi$CabinPos <- as.factor(combi$CabinPos)
combi$Deck <- as.factor(combi$Deck)
combi$Mother <- as.factor(combi$Mother)
combi$Child <- as.factor(combi$Child)
combi$FamilyID <- as.factor(combi$FamilyID)
combi$FamilyID2 <- as.factor(combi$FamilyID2)
combi$Title <- as.factor(combi$Title)
combi$Pclass <- as.factor(combi$Pclass)
combi$Embarked <- as.factor(combi$Embarked)
combi$Age <- as.numeric(combi$Age)

# Split back into test and train sets
train <- combi[1:891,]
test <- combi[892:1309,]
train$Survived<-as.factor(train$Survived)

sapply(train, function(x) sum(is.na(x)))


summary(train$Age) 
############################################### 
## RF MODEL
#Build Random Forest Ensemble
set.seed(415)

fit.rf <- randomForest(Survived ~ Pclass + Age + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID2 + Deck,
                    data=train, ntree=500,importance=T)
# Look at variable importance
varImpPlot(fit.rf)
plot(fit.rf,main='randomForest error rate')
imp<-importance(fit.rf,type='1')
imp<-imp[order(imp),]
imp
#write submission with rf method
Prediction.rf<-predict(fit.rf,test,OOB=TRUE,type='response')
submit.rf <- data.frame(PassengerId = test$PassengerId, Survived = Prediction.rf)

write.csv(submit.rf,'submission_randomForest.csv',row.names=F)


## CF MODEL
rf_ranges <- list(ntree = c(500, 1000, 1500, 2000), mtry = 3:8)
rf_tune <- tune(randomForest, Survived~Pclass + Age + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID2 + Deck, data=train, ranges=rf_ranges)

fit.cf <- cforest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + 
                 FamilySize + FamilyID + Deck + CabinPos,
               data = train, controls=cforest_unbiased(ntree=2000, mtry=3)) #ntree=2000, mtry=3
Prediction.cf <- predict(fit.cf, test, OOB=TRUE, type = "response")
plot(fit.cf)
plot(fit.cf,main='conditional inference tree error rate')
submit.cf <- data.frame(PassengerId = test$PassengerId, Survived = Prediction.cf)
write.csv(submit.cf, file = "submission_cforest.csv", row.names = FALSE)
