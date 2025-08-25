import numpy as np

xT = np.linspace(1,55,10)
tauF = 10.0
tauR = 20.0
frt = 0.75
testF = np.exp(-xT/tauF)
testR = (1-frt)*np.exp(-xT/tauR) + frt*tauR/(tauF-tauR)*(np.exp(-xT/tauF) -np.exp(-xT/tauR)) +frt*testF

deviation = 0.03
errF = np.random.uniform(low = -deviation, high= deviation, size=10)
errR = np.random.uniform(low = -deviation, high= deviation, size=10)

testF = testF +errF*testF
testR = testR +errR*testR

for i,dis in enumerate(xT):
    print(f'{dis}   {testR[i]} ')
for i,dis in enumerate(xT):
    print(f'{dis}   {testF[i]} ')

