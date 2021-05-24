###### Plot Sine waves for explanation Fourier Transform ######
rm(list = ls())

t=seq(0,4,0.0001)  #time
f1=2             #frequency
f2=5
A=1             #amplitude

y1=A*sin(((2*pi))*f1*t)
y2=A*sin(((2*pi))*f2*t)
z = y1+y2
df = read.csv("C:\\Users\\Jitte\\PycharmProjects\\Floods\\IBF_TriggerModel_Flood-rainfall_v12\\df_extra.csv", header=T, sep=",")
df_district = subset(df, df$district == 'bungoma')
data = df_district$dis_station1
time = df_district$time
plot(time, data, type = 'l', xlab = 'time in seconds', ylab = 'Signal', col = 2,
     yaxt = 'n', cex.lab = 1.4, cex.axis = 1.4)

library(rpatrec) #to add noise to a signal
#noised_y1 = noise(z, 'white', 0.5)
#plot(t, noised_y1, type = 'l', xlab = 'time in seconds', ylab = 'Signal', col = 2,
 #    yaxt = 'n', cex.lab = 1.4, cex.axis = 1.4)


#plot(t,z,type="l", xlab="time in seconds", ylab="Sine wave", col = 2,
 #    yaxt = 'n', cex.lab = 1.4, cex.axis = 1.4)


####### Plot frequency domain #####
library(spectral)

FT = spec.fft(data, time)
plot(FT,
  ylab = "Amplitude",
  xlab = "Frequency",
  type = "l",
  xlim = c(0, 10),
  col = 2,
  cex.lab = 1.4, cex.axis = 1.4
)
