from numpy import *
from pylab import plot, show
# first, create an arbitrary time series, ts
ts = [0]
for i in range(1,100000):
    ts.append(ts[i-1]*1.0 + random.randn())
# calculate standard deviation of differenced series using various lags
lags = range(2, 20)
tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
# plot on log-log scale
plot(log(lags), log(tau)); show()
# calculate Hurst as slope of log-log plot
m = polyfit(log(lags), log(tau), 1)
hurst = m[0]*2.0
print 'hurst = ',hurst