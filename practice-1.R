#---- 3장 의사결정나무 ----

# 3장 1번 문제

install.packages("rpart")
install.packages("rpart.plot") # 패키지 설치
head(iris, 3) # 데이터 확인

# 1차 분류나무 생성
library(rpart)
set.seed(1234)
pre.control <- rpart.control(xval=10, cp=0, minsplit=10)
tree.iris1 <- rpart(Species~., data=iris, method="class", control=pre.control)
print(tree.iris1)

# 분류나무 그림 확인
library(rpart.plot)
prp(tree.iris1, type=4, extra=1, digits = 2, box.palette = "gray")

# 예측값 확인
pred1 <- predict(tree.iris1, iris, type = "class")
tail(pred1)
tab1 <- table(iris$Species, pred1, dnn=c("Observed","Predicted"))
print(tab1)

# 모형 성능 평가
AccRate1 <- sum(diag(tab1))/sum(tab1)
ErrRate1 <- 1-AccRate1
print(cat("tree1 정분류율:", round(AccRate1, 3), " 오분류율:", round(ErrRate1, 3)))

# 가지치기 수행
cps <- printcp(tree.iris1)
k <- which.min(cps[, "xerror"]) # xerror 가 최소가 되는 k값
err = cps[k,"xerror"]; se = cps[k,"xstd"]
c = 1 # 1-s.e. 법칙
k1 = which(cps[,"xerror"] <= err+c*se)[1]
cp.chosen = cps[k1,"CP"]

tree.iris2 = prune(tree.iris1, cp=cp.chosen)
print(tree.iris2)

# 분류나무 그림 확인
prp(tree.iris2, type=4, extra=1, digits=2, box.palette="Grays")

# 노드 최소 크기 5로 재분할
re.control <- rpart.control(xval=10, cp=0, minsplit=5)
tree.iris3 <- rpart(Species~., data=iris, method="class", control=re.control)
print(tree.iris3)
prp(tree.iris3, type=4, extra=1, digits = 2, box.palette = "gray")

# 가지치기 수행
cps <- printcp(tree.iris3)
k <- which.min(cps[, "xerror"]) # xerror 가 최소가 되는 k값
err = cps[k,"xerror"]; se = cps[k,"xstd"]
c = 1 # 1-s.e. 법칙
k1 = which(cps[,"xerror"] <= err+c*se)[1]
cp.chosen = cps[k1,"CP"]

tree.iris4 = prune(tree.iris3, cp=cp.chosen)
print(tree.iris4)

# 분류나무 그림 확인
prp(tree.iris4, type=4, extra=1, digits=2, box.palette="Grays")

# 예측값 확인
pred2 <- predict(tree.iris4, iris, type = "class")
tail(pred2)
tab2 <- table(iris$Species, pred2, dnn=c("Observed","Predicted"))
print(tab2)

# 모형 성능 평가
AccRate2 <- sum(diag(tab2))/sum(tab2)
ErrRate2 <- 1-AccRate2
print(cat("tree4 정분류율:", round(AccRate2, 3), " 오분류율:", round(ErrRate2, 3)))

#---- 6장 모형비교평가 ----

getwd()

# 데이터 준비
library(MASS)
head(Boston)
str(Boston)

# 훈련데이터-검증데이터 분할
set.seed(1234)
train.index = sample(1:nrow(Boston), size=0.7*nrow(Boston), replace = F)
Boston.train = Boston[train.index, ] # 훈련데이터 70%
Boston.test = Boston[-train.index, ] # 검증데이터 30%

# 회귀모형 만들기
fit.reg = lm(medv~., data = Boston.train)
fit.step.reg = step(fit.reg, direction = "both", trace = FALSE)
summary(fit.step.reg)

# 회귀모형 검증
pred.reg = predict(fit.step.reg, newdata = Boston.test, type = "response")
head(pred.reg)
reg.mse = mean((Boston.test$medv-pred.reg)^2) # MSE
reg.mae = mean(abs(Boston.test$medv-pred.reg)) # MAE

# 의사결정나무 만들기
library(rpart)
my.cont = rpart.control(xval = 10, cp=0, minsplit = 5)
fit.tree = rpart(medv~., data = Boston.train, method = "anova", control = my.cont)
tmp = printcp(fit.tree) # cp값에 대한 나무구조 순서 출력
(k = which.min(tmp[, "xerror"])) # 최적의 나무 사이즈 (최소 오분류율)
(cp.tmp = tmp[k, "CP"]) # 교차타당성 방법에 의한 오분류율 최솟값
fit.prun.tree = prune(fit.tree, cp=cp.tmp) # 가지치기 (15번째 단계)
print(fit.prun.tree)

# 나무모형 검증
pred.tree = predict(fit.prun.tree, newdata = Boston.test, type = "vector")
(tree.mse = mean((Boston.test$medv-pred.tree)^2))
(tree.mae = mean(abs(Boston.test$medv-pred.tree)))

# 신경망모형 만들기
install.packages("neuralnet")
install.packages("dummy")
library(neuralnet)
library(dummy)

dvar = c(1:4)
Boston2 = dummy(x=Boston[,dvar]) # 더미변수 만들기
Boston2 = Boston2[, -c(5, 7, 13, 25)]
Boston2 = cbind(Boston[,-dvar], Boston2)
for(i in 1:ncol(Boston2)) if(!is.numeric(Boston2[, i]))
  Boston2[, i] = as.numeric(Boston2[, i])

#(nn.mse = mean((Boston.test$medv-pred.nn)^2))
#(nn.mae = mean(abs(Boston.test$medv-pred.nn)))


# 랜덤포레스트 적합
install.packages("randomForest")
library(randomForest)
fit.rf = randomForest(medv~., data = Boston.train, ntree=100, mtry=5, 
                      importance=T, na.action = na.omit)

# 랜덤포레스트 검증
pred.rf = predict(fit.rf, newdata = Boston.test, type = "response")
(rf.mse = mean((Boston.test$medv-pred.rf)^2))
(rf.mae = mean(abs(Boston.test$medv-pred.rf)))

# 모형별 MSE, MAE 비교표
mses = c(reg.mse, tree.mse, rf.mse)
maes = c(reg.mae, tree.mae, rf.mae)
compare.table = data.frame(mses, maes)
rownames(compare.table) = c("reg", "tree", "rf")
compare.table

# 모형 관측값-예측값 산점도 비교

rm(list = ls())