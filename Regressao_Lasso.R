######################################################
#    TRABALHO DE IAA005 – Estatística Aplicada II    #
#                  Regressao Lasso                   # 
######################################################

# Instalando os pacotes necessarios
#install.packages("plyr")
#install.packages("readr")
#install.packages("dplyr")
#install.packages("caret")
#install.packages("ggplot2")
#install.packages("repr")
#install.packages("glmnet")

# Carregando os pacotes necessarios
library(plyr)
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(repr)
library(glmnet)

# Carregar base de dados
load("/home/forno/Documents/IAA/EST2/trabalhosalarios.RData")

# Configurar seed para padronizar os mesmos resultados
set.seed(123)  

# Criar os indices de particionamento para o dataset em 80% para treinamento
index = sample(1:nrow(trabalhosalarios),0.8*nrow(trabalhosalarios))

# Criar base de treinamento (80%)
train = trabalhosalarios[index,]  

# Criar a base de teste (20%)
test = trabalhosalarios[-index,] 


# Padronizar as variaveis nao binarias das bases de treinamento e teste
cols = c('husage', 'husearns', 'huseduc', 'hushrs',
         'age', 'educ', 'exper', 'lwage')
pre_proc_val <- preProcess(train[,cols], 
                           method = c("center", "scale"))
train[,cols] = predict(pre_proc_val, train[,cols])
test[,cols] = predict(pre_proc_val, test[,cols])

#######
# Selecionar variaveis para usar no modelo
cols_reg = c('husage', 'husunion', 'husearns', 'huseduc', 'husblck', 
             'hushisp', 'hushrs', 'kidge6', 'age', 
             'black', 'educ', 'hispanic', 'union', 'exper',
             'kidlt6', 'lwage')

# Gerar variaveis dummies para organizar os datasets em matrizes
dummies <- dummyVars(lwage~husage+husunion+husearns+huseduc+husblck+
                       hushisp+hushrs+kidge6+age+
                       black+educ+hispanic+union+exper+
                       kidlt6, 
                     data = trabalhosalarios[,cols_reg])
train_dummies = predict(dummies, newdata = train[,cols_reg])
test_dummies = predict(dummies, newdata = test[,cols_reg])

# Guardar a matriz de dados de treinamentos das variaveis explicativas
x_train = as.matrix(train_dummies)
# Guardar o vetor de dados de treinamentos da variavel dependente
y_train = train$lwage
# Guardar a matriz de dados de teste das variaveis explicativas
x_test = as.matrix(test_dummies)
# Guardar o vetor de dados de teste da variavel dependente
y_test = test$lwage

# Calcular o valor otimo de lambda
lambdas <- 10^seq(2, -3, by = -.1)
lasso_lamb <- cv.glmnet(x_train, y_train, alpha = 1, 
                        lambda = lambdas, 
                        standardize = TRUE, nfolds = 5)
best_lambda_lasso <- lasso_lamb$lambda.min 
best_lambda_lasso
# Lambda encontrado: 0,01

### Estimar o modelo Lasso
lasso_model <- glmnet(x_train, y_train, alpha = 1, 
                      lambda = best_lambda_lasso, 
                      standardize = TRUE)

# Visualizar resultado da estimativa dos coeficientes
lasso_model[["beta"]]

# Predicao nos dados de treinamento
predictions_train <- predict(lasso_model, 
                             s = best_lambda_lasso,
                             newx = x_train)

# Predicao nos dados de teste
predictions_test <- predict(lasso_model, 
                            s = best_lambda_lasso, 
                            newx = x_test)

######
# Criar funcao de calculo das metricas
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

# Calcular as metricas da base de treinamento
eval_results(y_train, predictions_train, train)
# RMSE       Rsquare
# 0.8388284  0.2960251

# Calcular as metricas da base de teste
eval_results(y_test, predictions_test, test)
# RMSE       Rsquare
# 0.8610461  0.2324235

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

# Construir matriz para a predicao
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

### Realizar predicao
predict_our_lasso <- predict(lasso_model, 
                             s = best_lambda_lasso, 
                             newx = our_pred)
predict_our_lasso
#-0.1685626

# Converter o resultado para valor nominal
wage_pred_lasso=(predict_our_lasso*
                   pre_proc_val[["std"]][["lwage"]])+
  pre_proc_val[["mean"]][["lwage"]]
wage_pred_lasso
#2.105575

# Aplicando antilog
exp(wage_pred_lasso)
#8.211825

######
### Calcular o intervalo de confianca
n <- nrow(train)
m <- wage_pred_lasso
s <- pre_proc_val[["std"]][["lwage"]]
dam <- s/sqrt(n)
CIlwr_lasso <- m + (qnorm(0.025))*dam
CIupr_lasso <- m - (qnorm(0.025))*dam 

exp(CIlwr_lasso)
exp(CIupr_lasso)
# Segundo as caracteristicas que atribuimos o salario-hora da esposa eh em media U$8.211825 e pode
# variar entre U$8.028111 e U$8.399743

#############################################################