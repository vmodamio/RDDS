import itertools
import numpy as np
import scipy.optimize
from ispline import Isplines
import matplotlib.pyplot as plt

order = 5
mesh = [0.0,  0.2, 0.4, 0.6, 0.8, 1.0]

xplot = np.linspace(0, 1, 1000)

isplines_xplot = Isplines(order, mesh, xplot)

data = {'x': xplot}
for i in range(1, isplines_xplot.n +1):
    data[f"I{i}"] = isplines_xplot.I(i)
    plt.plot(xplot, data[f'I{i}'])

plt.show()

