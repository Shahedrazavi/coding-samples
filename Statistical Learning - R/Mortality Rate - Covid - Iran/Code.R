Sys.setlocale(locale = 'persian')


overall_p <- function(my_model) {
  f <- summary(my_model)$fstatistic
  p <- pf(f[1],f[2],f[3],lower.tail=F)
  attributes(p) <- NULL
  return(p)
}

library(data.table)
library(ggplot2)

d = fread('F:/University/01-02/Term6/Statistical Learning (Regression)/Homework/PS1/iranprovs_mortality_monthly.csv', encoding = 'UTF-8')

class(d)

names(d)
summary(d)

d$ym_num = d$y + d$m / 12 - 1/24

# ds = d[, .(n = sum(n)), .(y, m, ym_num)]
# ggplot(ds, aes(ym_num, n))+
#   geom_line()+
#   geom_point()+
#   scale_x_continuous(breaks = 1389:1401)+
#   scale_y_continuous(limits = c(10000, 75000))
# 
# max(ds$n)


ds = d[, .(n = sum(n)), .(y, m, ym_num, prov)]

provs = length(unique(ds$prov))
months = length(unique(ds$m))

ym_num_covid = 1398 + 10/12 - 1/24
ym_num_start = ym_num_covid - 5


### Data frame that the results are appended to it
result = data.frame(y=numeric(),
                    m=numeric(),
                    ym_num=numeric(),
                    prov=character(),
                    n=numeric(),
                    n_predicted=numeric(),
                    excess_mort=numeric(),
                    ratio=numeric(),
                    stringsAsFactors=FALSE)

result_2 = data.frame(y=numeric(),
                    m=numeric(),
                    ym_num=numeric(),
                    prov=character(),
                    n=numeric(),
                    n_predicted=numeric(),
                    excess_mort=numeric(),
                    ratio=numeric(),
                    stringsAsFactors=FALSE)




# prv = unique(ds$prov)[8]
# mo = 9


# Iterating through all month-year and province combinations #
for (prv in 1:provs){
 for (mo in 1:months){

   # print(prv)
   # print(mo)
   dat = ds[prov == unique(ds$prov)[prv] & m == mo,]

   dat = dat[ym_num > ym_num_start]

   dat_pre = dat[ym_num < ym_num_covid]
   dat_post= dat[ym_num > ym_num_covid]

   fit = lm(n ~ ym_num, dat_pre)
   summary(fit)


   p_val = overall_p(fit)
   p_val
   if (p_val>0.1){
     pred = mean(dat_pre$n)
     dat$n_predicted = pred
     dat_post$n_predicted = pred
   }else{
     dat$n_predicted = predict(fit, dat)
     dat_post$n_predicted = predict(fit, dat_post)
   }
   
   dat_post$excess_mort = dat_post$n - dat_post$n_predicted
   dat_post$ratio = dat_post$excess_mort / dat_post$n

   sqrt(var(dat_pre$n))

   for (k in 1:dim(dat_post)[1]){
     result_2 = rbindlist(list(result_2, dat_post[k:k,]))
     if(dat_post$n[k] >= dat_post$n_predicted[k]+2*sqrt(var(dat_pre$n))){
       result = rbindlist(list(result, dat_post[k:k,]))
     }
   }
 }
}

ggplot(result, aes(ym_num,prov,fill=ratio))+
  geom_tile()


## Calculating covid Impact and excess mortality for each province ##
result_prov = result[, .(excess_mort = sum(excess_mort)), .(prov)]


## Calculating covid Impact and excess mortality in Iran ##
sum(result$excess_mort)


# The more the number of improvemnent for each province, the better improvement in controlling covid #
improve_prov = data.frame(prov=character(),
                          ratio_1=numeric(),
                          ratio_2=numeric(),
                          improvement=numeric(),
                          stringsAsFactors=FALSE)

res = result_2[(y == 1401 | y== 1400) & m==6,]
for (x in provs){
  res_prov = res[prov == unique(ds$prov)[x],]
  print(res_prov)
  print(res_prov[1][8])
  # print(res[])
}


write.csv(result, "F:\\University\\01-02\\Term6\\Statistical Learning (Regression)\\Homework\\PS1\\result.csv", row.names=FALSE, fileEncoding = "UTF-8")
write.csv(result_2, "F:\\University\\01-02\\Term6\\Statistical Learning (Regression)\\Homework\\PS1\\result_2.csv", row.names=FALSE, fileEncoding = "UTF-8")
write.csv(result_prov, "F:\\University\\01-02\\Term6\\Statistical Learning (Regression)\\Homework\\PS1\\result_prov.csv", row.names=FALSE, fileEncoding = "UTF-8")
          