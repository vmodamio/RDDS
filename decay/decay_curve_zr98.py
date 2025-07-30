import matplotlib.pyplot as plt
import numpy as np

decayT, decayR, decayRerr = np.loadtxt('decay_98zr.dat', unpack= True, delimiter=None)
diffT, diffR = np.loadtxt('derivative_98zr.dat', unpack= True, delimiter=None)

diffRerr = 0.02*np.ones(diffR.shape)

plt.errorbar(decayT, decayR, yerr=decayRerr, fmt='o', elinewidth=1, color='black', capsize=4)
plt.errorbar(diffT, diffR, yerr=diffRerr, fmt='o', elinewidth=1, color='red', capsize=4)


plt.xlabel('Time (ps)')
plt.ylabel('R(t)')

plt.show()

