##########################################################
# Credit ScoreCard Model
# Kate Zhao
# Feb 2019
# Data from: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/ 
##########################################################

# ---------- Import libraries
library(utils)
library(caret)
library(ROCR)
library(ModelMetrics)
library(ggplot2)
library(randomForest)

# ---------- Import data
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
data = read.table(url, header = FALSE, sep = ' ', 
                  col.names = c("ChkAcctStat", "DurationMon", "CreditHist", "Purpose", "CrdtAmt",
                                "SavAcctBonds", "PresEmploy", "InstallRate", "PerStatus", "DebtorsGuarantors",
                                "PresRes", "Property", "AgeYrs", "OtherInstallPlans", "Housing", 
                                "NoExstCrdt", "Job", "NoPplMaintLiable", "Phone", "ForeignWker", 
                                "GoodBad"))

data$GoodBad <- as.factor(ifelse(data$GoodBad == 2, "Bad", "Good"))

# ---------- Visualizations
# Credit History (by GvB)
tab <- as.data.frame(table(data$CreditHist, data$GoodBad))
names(tab) <- c("CreditHist", "GoodBad", "Counts")
ggplot(data = tab, aes(x = CreditHist, y = Counts, fill=GoodBad)) + 
  geom_bar(stat = "identity") + scale_fill_grey() + ggtitle("Credit History by GoodvBad")

# Credit Amount
qplot(data$CrdtAmt, geom = "histogram", binwidth=500, main = "Credit Amount Distribution", 
      xlab = "Credit Amount", ylab = "Count")

# ---------- Definitions and models

# --- Stepwise Model
# Split data into test and train (70:30) WHY THIS RATIO?
set.seed(252)
split <- createDataPartition(y = data$GoodBad, p = 0.7, list = FALSE)
training <- data[split,]
testing <- data[-split,]

# Stepwise regression
modelstart <- glm(GoodBad ~., training, family = binomial)
modelstep <- step(modelstart, trace = FALSE, steps = 5000, k = log(nrow(training)))
summary(modelstep)

# Variable importance
varImp(modelstep, scale = FALSE)

# Plot ROCR and calculate AUC
prstep <- predict(modelstep, newdata = testing, type = "response")
predstep <- prediction(prstep, testing$GoodBad)
perfstep <- performance(predstep, "tpr", "fpr")
plot(perfstep, colorize = TRUE, main = "ROC Curve", col = 2, lwd = 2, 
     print.cutoffs.at = seq(0, 1, by=0.1), text.adj=c(-0.2, 1.7))
abline(a = 0, b = 1, lwd = 2, lty = 2, col = "gray")

round(auc(testing$GoodBad, prstep), digits = 2)

# --- Cross Validation
set.seed(123)
control <- trainControl(method = "repeatedcv", repeats = 3, classProbs = TRUE, summaryFunction = twoClassSummary)
fitCV <- train(GoodBad ~ ., data = training, method = "glm", family = binomial, 
               tuneLength = 5, trControl = control, metric = "ROC")
summary(fitCV)

# Variable importance
varImp(fitCV, scale = FALSE)

# Plot ROCR and calculate AUC
prCV = predict(fitCV, newdata = testing, type = "prob")
predCV = prediction(prCV[,2], testing$GoodBad)
perfCV = performance(predCV, "tpr", "fpr")
plot(perfCV, colorize = TRUE, main = "ROC Curve", col = 2, lwd = 2, 
     print.cutoffs.at = seq(0, 1, by=0.1), text.adj = c(-0.2, 1.7))
abline(a = 0, b = 1, lwd = 2, lty = 2, col = "gray")

round(auc(testing$GoodBad, prCV[,2]), digits = 2)

# --- Decision Tree

# --- Bagging

# --- Random Forest
rfmodel <- randomForest(GoodBad ~., training, importance = TRUE)

# ---------- Calculating Credit Score


# ---------- Notes
# FICO Score Elements: payment history (35%), credit utilization (30%), length of credit history (15%), new credit (10%), credit mix (10%)
# https://www.creditcards.com/credit-card-news/help/5-parts-components-fico-credit-score-6000.php

# https://rpubs.com/bkarun/228103 

