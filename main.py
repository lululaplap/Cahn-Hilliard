import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from periodic_lattice import Periodic_Lattice
import scipy.ndimage.filters as filter
from scipy import signal
import copy
fig = plt.figure()
#fig2 = plt.figure()

ax1 = fig.add_subplot(1,1,1)
#ax2 = fig.add_subplot(2,1,2)



class Cahn():
    def __init__(self,N):
        self.N = N
        p=0.5
        self.M = 0.1
        self.dx = 0.01
        self.dt = self.dx**2
        self.k = 0.1
        self.a = 0.1
        self.norm = self.M*self.dt/(self.dx**2)
        print(self.norm)
        self.grad = 0.5*np.array([[0,-1,0],[-1,2,-1],[0,-1,0]])
        self.laplace = np.array([[0,1,0],[1,-4,1],[0,1,0]])
        print(self.laplace)
        #phi = np.ones((self.N,self.N))*0.5+np.random.rand(self.N,self.N)-np.random.rand(self.N,self.N)
        #phi = np.zeros((self.N,self.N))+np.random.rand(self.N,self.N)/100-np.random.rand(self.N,self.N)/100
        phi = np.random.uniform(-0.01,0.01,size=(self.N,self.N))
        #phi = np.random.choice(a=[-0.5,0.5], size=(N, N), p=[p, 1-p])
        self.phi = phi
        self.F = []

    def getMu(self):
        phi_old = copy.deepcopy(self.phi)
        l = signal.convolve2d(phi_old,self.laplace, mode="same",boundary = "wrap")
        a = -self.a*phi_old
        b =  self.a*(np.power(phi_old,3))
        c =  -self.k*self.norm*l
        t = a+b+c
        return t
    def getPhi(self):
        mu = self.getMu()
        l = signal.convolve2d(mu,self.laplace, mode="same",boundary = "wrap")
        phi = self.phi + (self.norm * l)
        return phi

    def getf(self):
        g = signal.convolve2d(self.phi,self.grad, mode="same",boundary = "wrap")
        return -1*((self.a/2)*np.power(self.phi,2)) + self.a/4*np.power(self.phi,4) + (self.k/2 * self.norm* (np.power(g,2)))
    def update(self,i):


        for i in range(1000):
            self.phi = self.getPhi()
            self.F.append(np.sum(self.getf()))

        ax1.clear()
        ax1.imshow(self.phi, interpolation='sinc', cmap = 'cool', vmin=-1, vmax=1, origin='lower')
        plt.title(i)
        #ax2.clear()
        #ax2.plot(self.F)
        #lt.axis(option='square')

        #return self.phi

def main():
    C = Cahn(50)
    ani = animation.FuncAnimation(fig, C.update)
    plt.show()
    plt.plot(C.F)
    plt.show()
main()
