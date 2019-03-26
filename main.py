import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from periodic_lattice import Periodic_Lattice
import scipy.ndimage.filters as filter

fig = plt.figure()
#fig2 = plt.figure()

ax1 = fig.add_subplot(1,1,1)
ax2 = fig.add_subplot(2,1,1)


class Cahn():
    def __init__(self,N):
        self.N = N
        p=0.5
        self.M = 0.1
        self.dt = 0.01
        self.dx = 0.5
        self.k = 0.1
        self.a = 0.1
        self.norm = self.M*self.dt/(self.dx**2)
        print(self.norm)
        #phi = np.ones((self.N,self.N))*0.5+np.random.rand(self.N,self.N)-np.random.rand(self.N,self.N)
        phi = np.zeros((self.N,self.N))+np.random.rand(self.N,self.N)/100-np.random.rand(self.N,self.N)/100
        #phi = np.random.choice(a=[-0.5,0.5], size=(N, N), p=[p, 1-p])
        self.phi = Periodic_Lattice(phi)
        self.F = []

    def getMu(self):
        l = filter.laplace(self.phi,mode='wrap')
        return -self.a*self.phi+self.a*(self.phi**3)-self.k*self.norm*l
    def getPhi(self):
        mu = self.getMu()
        phi = self.phi +  self.norm * filter.laplace(mu)
        return phi

    def getf(self):
        xg,yg = np.gradient(self.phi)
        return -1*(self.a/2*self.phi**2) + self.a/4*self.phi**4 + self.k/2 *(xg**2+yg**2)

    def update(self,i):
        newPhi = self.getPhi()
        self.phi = newPhi
        self.F.append(np.sum(self.getf()))
        ax1.clear()
        ax1.imshow(self.phi, interpolation='sinc',
cmap='coolwarm', vmin=-1, vmax=1, origin='lower')
        plt.title(i)
        ax2.clear()
        ax2.plot(self.F)

        return self.phi

def main():
    C = Cahn(50)
    ani = animation.FuncAnimation(fig, C.update)
    plt.show()
    plt.plot(C.F)
    plt.show()
main()
