library(readxl)
data <- read_excel("Yuanchang.xlsx")
data$Date <- as.Date(data$time)
library(Rbeast)

# data$Yuanchang_2
result <- beast(data$Yuanchang_2)
imputed_series <- result$trend$Y + result$season$Y
data$Yuanchang_2 <- imputed_series

# csv
write.xlsx(data, "Yuanchang_imputed.xlsx", row.names=FALSE)
