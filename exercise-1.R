# 로지스틱회귀모형 연습문제

# 데이터 생성
q2_data <- read.csv("/winequalityCLASS.csv", header = T)
head(q2_data)

# alcohol 1개만 입력변수로 사용
fit.al <- glm(quality~alcohol, family = binomial, data = q2_data)
summary(fit.al)

# sulphates 1개만 입력변수로 사용
fit.sul <- glm(quality~sulphates, family = binomial, data = q2_data)
summary(fit.sul)

# 일부 변수를 선택해 입력변수로 사용
fit.all <- glm(quality~., family = binomial, data = q2_data)
fit.step <- step(fit.all, direction = "both") # 단계적 변수 선택
fit.step$anova
summary(fit.step)

# 모형 평가
# alcohol 변수만 선택한 모형
p.al <- predict(fit.al, newdata=q2_data, type="response") # 예측값 계산
cutoff = 0.5
yhat.al = ifelse(p.al > cutoff, 1, 0) # 확률값을 범주화
tab.al <- table(q2_data$quality, yhat.al, dnn=c("Observed","Predicted")) 
print(tab.al) # 분할표 작성

# sulphates 변수만 선택한 모형
p.sul <- predict(fit.sul, newdata=q2_data, type="response") # 예측값 계산
cutoff = 0.5
yhat.sul = ifelse(p.sul > cutoff, 1, 0) # 확률값을 범주화
tab.sul<- table(q2_data$quality, yhat.sul, dnn=c("Observed","Predicted")) 
print(tab.sul) # 분할표 작성

# 단계적 변수 선택 모형
p.step <- predict(fit.step, newdata=q2_data, type="response") #예측값 계산
cutoff = 0.5
yhat.step = ifelse(p.step > cutoff, 1, 0) # 확률값을 범주화
tab.step<-table(q2_data$quality, yhat.step,dnn=c("Observed","Predicted")) 
print(tab.step) # 분할표 작성

# 세 가지 모형 성능 비교표 작성
accuracy = c(sum(diag(tab.al))/sum(tab.al),
             sum(diag(tab.sul))/sum(tab.sul), 
             sum(diag(tab.step))/sum(tab.step))
sensitivity = c(tab.al[2,2]/sum(tab.al[2,]),
                tab.sul[2,2]/sum(tab.sul[2,]),
                tab.step[2,2]/sum(tab.step[2,]))
specificity = c(tab.al[1,1]/sum(tab.al[1,]),
                tab.sul[1,1]/sum(tab.sul[1,]),
                tab.step[1,1]/sum(tab.step[1,]))
tab.eval <- data.frame(accuracy, sensitivity, specificity,
                       row.names = c("eval.al", "eval.sul", "eval.step"))
print(tab.eval)


## alcohol 모형
## ROC(p.al, q2_data$quality, plot="ROC", AUC=T, main="logistic regression")
## sulphates 모형
## ROC(p.sul, q2_data$quality, plot="ROC", AUC=T, main="logistic regression")
## 단계적 선택 모형
## ROC(p.step, q2_data$quality, plot="ROC", AUC=T, main="logistic regression")


# 분류나무 연습문제

# 데이터 생성
q3_data <- read.csv("C:/Users/bluei/Rstudio_data/mid_dm1.csv", header = T)
head(q3_data)

# 데이터 범주화
q3_data$X1 <- factor(q3_data$X1)
q3_data$X2 <- factor(q3_data$X2)
q3_data$Y <- factor(q3_data$Y)
str(q3_data)

library(rpart)
library(rpart.plot)
set.seed(1234)

# 분류의사결정나무 생성
# 지니지수 이용
my.control <- rpart.control(xval = 5, minsplit = 1, cp=0, maxcompete = 4)
part <- rpart(Y~., data = q3_data, method = "class", control = my.control)
print(part)

# 의사결정나무 시각화
prp(part, type = 4, extra = 1, box.palette = "Grays")


plotcp(part) # 분할 시 cp값 확인
my1.control <- rpart.control(xval = 5, minsplit = 1, cp=0.16) # cp=0.16 이용

part1 <- rpart(Y~., data = q3_data, method = "class", control = my1.control)
print(part1) # 분류의사결정나무 확인
prp(part1, type = 4, extra = 1, box.palette = "Grays") # 시각화


rm(list = ls())