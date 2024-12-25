n = 100

library(data.table)
library(ggplot2)

ds = data.frame(name = sample(LETTERS, n, replace = T),
               course = paste0(sample(letters, n, replace = T),
                               sample(letters, n, replace = T)),
               score = runif(n, 6, 20))
ds[, -2]

duplicated(ds[, -3])

ds = data.table(ds)

#ds = ds[1:4, .(course, score)]

ds[score < 10, .(name, course), ]  

ds_names = ds[ , .(score_avg = mean(score),
                   score_sd = sd(score)), 
               .(name)]
