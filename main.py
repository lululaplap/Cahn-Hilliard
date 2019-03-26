import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from periodic_lattice import Periodic_Lattice
import scipy.ndimage.filters as filter

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)


class Cahn():
    def __init__(self,N):
        self.N = N
        p=0.5
        self.M = 1
        self.dt = 0.01
        self.dx = 0.5
        self.k = 0.1
        self.a = 0.1
        self.norm = self.M*self.dt/(self.dx**2)
        phi = np.random.choice(a=[-1,1], size=(N, N), p=[p, 1-p])
        self.phi = Periodic_Lattice(phi)

    def getMu(self):
        l = filter.laplace(self.phi,mode='wrap')
        return -self.a*self.phi+self.a*self.phi**2-self.k*self.norm*l
    def getPhi(self):
        mu = self.getMu()
        phi = self.phi +  self.norm * filter.laplace(mu)
        return phi
    def update(self,i):
        newPhi = self.getPhi()
        self.phi = newPhi
        ax1.clear()
        ax1.imshow(self.phi*100)
        return self.phi

def main():
    C = Cahn(50)
    for i in range(10):
        C.update(i)
    ani = animation.FuncAnimation(fig, C.update)
    plt.show()
main()
