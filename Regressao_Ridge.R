######################################################
#    TRABALHO DE IAA005 – Estatística Aplicada II    #
#                  Regressao Ridge                   # 
######################################################

#Instalar pacotes
#install.packages("plyr")
#install.packages("caret")
#install.packages("ggplot2")
#install.packages("glmnet")

# Carregar pacotes
library(plyr)
library(caret)
library(ggplot2)
library(glmnet)

# Carregar base de dados
load("/home/forno/Documents/IAA/EST2/trabalhosalarios.RData")

# Configurar seed para padronizar os mesmos resultados
set.seed(123)  

# Criar os indices de particionamento para o dataset em 80% para treinamento
index = sample(1:nrow(trabalhosalarios), 0.8 * nrow(trabalhosalarios))

# Criar base de treinamento (80%)
train = trabalhosalarios[index,]  

# Criar a base de teste (20%)
test = trabalhosalarios[-index,] 

# Padronizar as variaveis nao binarias das bases de treinamento e teste
cols = c('husage', 'husearns', 'huseduc', 'hushrs', 'earns',
         'age', 'educ', 'exper', 'lwage')
pre_proc_val <- preProcess(train[,cols], method = c("center", "scale"))
train[,cols] = predict(pre_proc_val, train[,cols])
test[,cols] = predict(pre_proc_val, test[,cols])

# Selecionar variaveis para usar no modelo
# **Tirado 'earns' pq nao tem na predicao pedida no trabalho
cols_reg = c('husage', 'husunion', 'husearns', 'huseduc', 'husblck', 
             'hushisp', 'hushrs', 'kidge6', 'age', 
             'black', 'educ', 'hispanic', 'union', 'exper',
             'kidlt6', 'lwage')

# Gerar variaveis dummies para organizar os datasets em matrizes
# **Tirado 'earns' pq nao tem na predicao pedida no trabalho
dummies <- dummyVars(lwage~husage+husunion+husearns+huseduc+husblck+
                       hushisp+hushrs+kidge6+age+
                       black+educ+hispanic+union+exper+
                       kidlt6, 
                     data = trabalhosalarios[,cols_reg])
train_dummies = predict(dummies, newdata = train[,cols_reg])
test_dummies = predict(dummies, newdata = test[,cols_reg])

x_train = as.matrix(train_dummies)
y_train = train$lwage

x_test = as.matrix(test_dummies)
y_test = test$lwage

# Calcular o valor otimo de lambda
lambdas <- 10^seq(2, -3, by = -.1)
ridge_lamb <- cv.glmnet(x_train, y_train, alpha = 0, lambda = lambdas)
best_lambda_ridge <- ridge_lamb$lambda.min

### Estimar o modelo Ridge
start <- Sys.time()
ridge_reg = glmnet(x_train, y_train, nlambda = 25, alpha = 0, 
                   family = 'gaussian', 
                   lambda = best_lambda_ridge)
end <- Sys.time()
difftime(end, start, units="secs")

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

# Predicao e avaliacao nos dados de treinamento
predictions_train <- predict(ridge_reg, 
                             s = best_lambda_ridge,
                             newx = x_train)

# Calcular as metricas da base de treinamento
eval_results(y_train, predictions_train, train)
# RMSE       Rsquare
# 0.8439961  0.2873244

# Predicao e avaliacao nos dados de teste
predictions_test <- predict(ridge_reg, 
                            s = best_lambda_ridge, 
                            newx = x_test)

# Calcular as metricas da base de teste
eval_results(y_test, predictions_test, test)
# RMSE       Rsquare
# 0.8032132  0.2661383

### Criar os dados para a predicao
#(anos – idade do marido)
husage = (40 - pre_proc_val[["mean"]][["husage"]]) / pre_proc_val[["std"]][["husage"]]
#(marido não possui união estável)
husunion = 0
#(US$ renda do marido por semana)
husearns = (600 - pre_proc_val[["mean"]][["husearns"]]) / pre_proc_val[["std"]][["husearns"]]
#(anos de estudo do marido)
huseduc = (13 - pre_proc_val[["mean"]][["huseduc"]]) / pre_proc_val[["std"]][["huseduc"]]
#(o marido é preto)
husblck = 1
#(o marido não é hispânico)
hushisp = 0
#(horas semanais de trabalho do marido)
hushrs = (40 - pre_proc_val[["mean"]][["hushrs"]]) / pre_proc_val[["std"]][["hushrs"]]
#(possui filhos maiores de 6 anos)
kidge6 = 1
# 
#earns = (? - pre_proc_val[["mean"]][["earns"]]) / pre_proc_val[["std"]][["earns"]]
#(anos – idade da esposa)
age = (38 - pre_proc_val[["mean"]][["age"]]) / pre_proc_val[["std"]][["age"]]
#(a esposa não é preta)
black = 0
#(anos de estudo da esposa)
educ = (13 - pre_proc_val[["mean"]][["educ"]]) / pre_proc_val[["std"]][["educ"]]
#(a esposa é hispânica)
hispanic = 1
#(esposa não possui união estável)
union = 0
#(anos de experiência de trabalho da esposa)
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
predict_our_ridge <- predict(ridge_reg, 
                             s = best_lambda_ridge, 
                             newx = our_pred)
predict_our_ridge
#-0.1870271

# Converter o resultado para valor nominal
wage_pred_ridge=(predict_our_ridge*
                   pre_proc_val[["std"]][["lwage"]])+
  pre_proc_val[["mean"]][["lwage"]]

wage_pred_ridge
#2.094653

# Aplicando antilog
exp(wage_pred_ridge)
#8.122623

### Calcular o intervalo de confianca
n <- nrow(train) # tamanho da amostra
m <- wage_pred_ridge # valor medio predito
s <- pre_proc_val[["std"]][["lwage"]] # desvio padrao
dam <- s/sqrt(n) # distribuicao da amostragem da media
CIlwr_ridge <- m + (qnorm(0.025))*dam # intervalo inferior
CIupr_ridge <- m - (qnorm(0.025))*dam # intervalo superior

exp(CIlwr_ridge)
exp(CIupr_ridge)

# Segundo as caracteristicas que atribuimos o salario-hora da esposa eh em media US$8.122623 e pode
# variar entre US$7.939315 e US$8.310162