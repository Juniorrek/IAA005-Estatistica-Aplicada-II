##############################################################
#       TRABALHO DE IAA005 – Estatística Aplicada II         #
#               Secao 2.4: Regressao ElasticNet              # 
##############################################################

# Instalando os pacotes necessarios
# install.packages("plyr")
# install.packages("readr")
# install.packages("dplyr")
# install.packages("ggplot2")
# install.packages("repr")
# install.packages("glmnet")
# install.packages("caret")

# Carregando os pacotes necessarios
library(plyr)
library(readr)
library(dplyr)
library(ggplot2)
library(repr)
library(glmnet)
library(caret)

# Carregar base de dados
setwd("D:/Dev/iaa-ufpr/disciplinas/IAA005 - Estatística Aplicada II/trabalho")
load("trabalhosalarios.RData")

# Configurar seed para padronizar os mesmos resultados
set.seed(123)

# Criar os indices de particionamento para o dataset em 80% para treinamento
index = sample(1:nrow(trabalhosalarios),0.8*nrow(trabalhosalarios))

# Criar base de treinamento (80%)
train = trabalhosalarios[index,]  

# Criar base de teste (20%)
test = trabalhosalarios[-index,] 

# Padronizar as variaveis nao binarias das bases de treinamento e teste
cols = c('husage', 'husearns', 'huseduc', 'hushrs',
         'age', 'educ', 'exper', 'lwage')

pre_proc_val <- preProcess(train[,cols], 
                           method = c("center", "scale"))
train[,cols] = predict(pre_proc_val, train[,cols])
test[,cols] = predict(pre_proc_val, test[,cols])

#summary(train)
#summary(test)

# Selecionar variaveis para usar no modelo
cols_reg = c('husage', 'husunion', 'husearns', 'huseduc', 'husblck', 
             'hushisp', 'hushrs', 'kidge6', 'age', 
             'black', 'educ', 'hispanic', 'union', 'exper',
             'kidlt6', 'lwage')

# Vamos gerar variaveis dummies para organizar os datasets
# em objetos tipo matriz
# Estamos interessados em estimar o salario-hora (lwage)
dummies <- dummyVars(lwage~husage+husunion+husearns+huseduc+husblck+
                       hushisp+hushrs+kidge6+age+
                       black+educ+hispanic+union+exper+
                       kidlt6, 
                     data = trabalhosalarios[,cols_reg])
train_dummies = predict(dummies, newdata = train[,cols_reg])
test_dummies = predict(dummies, newdata = test[,cols_reg])
#print(dim(train_dummies)); print(dim(test_dummies))

# Guardar a matriz de dados de treinamentos das variaveis explicativas
x = as.matrix(train_dummies)
# Guardar o vetor de dados de treinamentos da variavel dependente
y = train$lwage
# Guardar a matriz de dados de teste das variaveis explicativas
x_test = as.matrix(test_dummies)
# Guardar o vetor de dados de teste da variavel dependente
y_test = test$lwage

# Vamos configurar o treinamento do modelo por 
# cross validation, com 10 folders, 5 repeticoes
# e busca aleatoria dos componentes das amostras
# de treinamento, o "verboseIter" eh soh para 
# mostrar o processamento.
train_cont <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 5,
                           search = "random",
                           verboseIter = TRUE)

# Vamos treinar o modelo
elastic_reg <- train(lwage~husage+husunion+husearns+huseduc+husblck+
                       hushisp+hushrs+kidge6+age+
                       black+educ+hispanic+union+exper+
                       kidlt6, 
                     data = train,
                     method = "glmnet",
                     tuneLength = 10,
                     trControl = train_cont)

# ERRO AO EXECUTAR O COMANDO ACIMA
# Warning message:
# In nominalTrainWorkflow(x = x, y = y, wts = weights, info = trainInfo,  :
#  There were missing values in resampled performance measures.


# O melhor parametro alpha escolhido eh:
elastic_reg$bestTune

# E os parametros sao:
elastic_reg[["finalModel"]][["beta"]]

# Vamos fazer as predicoes e avaliar a performance do
# modelo

# Vamos fazer as predicoes no modelo de treinamento:
predictions_train <- predict(elastic_reg, x)

# Vamos calcular o R^2 dos valores verdadeiros e 
# preditos conforme a seguinte funcao:
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  # As metricas de performace do modelo:
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
}

# As metricas de performance na base de treinamento
# sao:
eval_results(y, predictions_train, train) 
# RMSE  Rsquare
# 0.8388957 0.295912

# Vamos fazer as predicoes na base de teste
predictions_test <- predict(elastic_reg, x_test)

# As metricas de performance na base de teste sao:
eval_results(y_test, predictions_test, test)
# RMSE   Rsquare
# 0.8608574 0.2327599

########
### Criar os dados para a predicao
#(anos - idade do marido)
husage = (40 - pre_proc_val[["mean"]][["husage"]]) / pre_proc_val[["std"]][["husage"]]
#(marido nao possui uniao estavel)
husunion = 0
#(US$ renda do marido por semana)
husearns = (600 - pre_proc_val[["mean"]][["husearns"]]) / pre_proc_val[["std"]][["husearns"]]
#(anos de estudo do marido)
huseduc = (13 - pre_proc_val[["mean"]][["huseduc"]]) / pre_proc_val[["std"]][["huseduc"]]
#(o marido e preto)
husblck = 1
#(o marido nao e hispanico)
hushisp = 0
#(horas semanais de trabalho do marido)
hushrs = (40 - pre_proc_val[["mean"]][["hushrs"]]) / pre_proc_val[["std"]][["hushrs"]]
#(possui filhos maiores de 6 anos)
kidge6 = 1
# 
#earns = (? - pre_proc_val[["mean"]][["earns"]]) / pre_proc_val[["std"]][["earns"]]
#(anos - idade da esposa)
age = (38 - pre_proc_val[["mean"]][["age"]]) / pre_proc_val[["std"]][["age"]]
#(a esposa nao e preta)
black = 0
#(anos de estudo da esposa)
educ = (13 - pre_proc_val[["mean"]][["educ"]]) / pre_proc_val[["std"]][["educ"]]
#(a esposa e hispanica)
hispanic = 1
#(esposa nao possui uniao estavel)
union = 0
#(anos de experiencia de trabalho da esposa)
exper = (18 - pre_proc_val[["mean"]][["exper"]]) / pre_proc_val[["std"]][["exper"]]
#(possui filhos menores de 6 anos)
kidlt6 =  1


# Vamos construir uma matriz de dados para a predicao
our_pred = as.matrix(data.frame(husage=husage, 
                                husunion=husunion,
                                husearns=husearns,
                                huseduc=huseduc,
                                husblck=husblck,
                                hushisp=hushisp,
                                hushrs=hushrs,
                                kidge6=kidge6,
                                age=age,
                                black=black,
                                educ=educ,
                                hispanic=hispanic,
                                union=union,
                                exper=exper,
                                kidlt6=kidlt6))

# Vamos fazer a predicao com base nos parametros que
# selecionamos
predict_our_elastic <- predict(elastic_reg,our_pred)
predict_our_elastic
# -0.1676672

# Novamente, o resultado eh padronizado, nos temos que
# reverte-lo para o nivel dos valores originais do
# dataset, vamos fazer isso:
wage_pred_elastic=(predict_our_elastic*
                     pre_proc_val[["std"]][["lwage"]])+
  pre_proc_val[["mean"]][["lwage"]]
wage_pred_elastic
# 2.106044

# Aplicando antilog
exp(wage_pred_elastic)
# 8.215678

# Entao o salario-hora medio da esposa predito com base
# nas caracteristicas informadas eh US$9.28

# Vamos criar o intervalo de confianca para o nosso
# exemplo
n <- nrow(train)
m <- wage_pred_elastic
s <- pre_proc_val[["std"]][["lwage"]]
dam <- s/sqrt(n)
CIlwr_elastic <- m + (qnorm(0.025))*dam
CIupr_elastic <- m - (qnorm(0.025))*dam 

# Os valores minimo e maximo sao:
CIlwr_elastic
# 2.083418
exp(CIlwr_elastic)
# 8.031877
CIupr_elastic
# 2.12867
exp(CIupr_elastic)
# 8.403684

# Segundo as caracteristicas que atribuimos o salario-hora da esposa eh em media U$8.215678 e pode
# variar entre U$8.031877 e U$8.403684
