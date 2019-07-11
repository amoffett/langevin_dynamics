import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.constants

class brownian_dynamics_2d():
    def __init__(self):
        self.k_B = sp.constants.Boltzmann
        
    def system(self, n_particles = 1, gamma = 1, T = 300, \
               V = np.array([[-1, -1, -6.5, 0.7, -10], \
                             [0, 0, 11, 0.6, 0], \
                             [-10, -10, -6.5, 0.7, -4], \
                             [-200, -100, -170, 15, -120], \
                             [1, 0, -0.5, -1, .25], \
                             [0, 0.5, 1.5, 1, 1.25]])):
        self.n_particles = 1
        self.gamma = gamma
        self.T = T
        self.V = V
        
    def potential(self, X):
        try: 
            self.V
        except AttributeError: 
            print "Define your system first."
            exit(2)
        value = 0
        for i in range(self.V.shape[1]):
            value += self.V[3, i] * np.exp(self.V[0, i] * (X[0] - self.V[4, i])**2 + \
                                      self.V[1, i] * (X[0] - self.V[4, i]) * \
                                           (X[1] - self.V[5, i]) + self.V[2, i] * (X[1] - self.V[5, i])**2)
        return value
    
    def force(self, X, bound = 200):
        value = np.array([[0, 0]])
        for i in range(self.V.shape[1]):
            value[0, 0] += -1 * (2 * self.V[0, i] * (X[0] - self.V[4, i]) + self.V[1, i] * (X[1] - self.V[5, i])) \
            * self.V[3, i] * np.exp(self.V[0, i] * (X[0] - self.V[4, i])**2 + \
                                      self.V[1, i] * (X[0] - self.V[4, i]) * \
                                           (X[1] - self.V[5, i]) + self.V[2, i] * (X[1] - self.V[5, i])**2)
            value[0, 1] += -1 * (2 * self.V[2, i] * (X[1] - self.V[5, i]) + self.V[1, i] * (X[0] - self.V[4, i])) \
            * self.V[3, i] * np.exp(self.V[0, i] * (X[0] - self.V[4, i])**2 + \
                                      self.V[1, i] * (X[0] - self.V[4, i]) * \
                                           (X[1] - self.V[5, i]) + self.V[2, i] * (X[1] - self.V[5, i])**2)
        if X[0] < -1:
            value[0, 0] += bound * (X[0] + 1)**2 
        elif X[0] > 4:
            value[0, 0] += -1 * bound * (X[0] - 4)**2
        if X[1] < -1.5:
            value[0, 1] += bound * (X[1] + 1.5)**2
        elif X[1] > 5.5:
            value[0, 1] += -1 * bound * (X[1] - 5.5)**2
        return value[0]
    
    def random_force(self, dt = 1):
        mean = np.zeros([2])
        cov = np.zeros([2, 2])
        np.fill_diagonal(cov, np.sqrt(2 * (dt * self.k_B * self.T) / self.gamma))
        value = np.random.multivariate_normal(mean, cov, 1)
        return value[0]
    
    def plot_potential(self, minx = -3, maxx = 1.5, miny = -1, maxy = 3):
        grid_width = max(maxx - minx, maxy - miny) / 200.0
        xx, yy = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
        V = self.potential([xx, yy])
        plt.contourf(xx, yy, V.clip(max = -self.V[3, :].min()), 40)
        plt.xlabel(r'X')
        plt.ylabel(r'Y')
        cbar = plt.colorbar()
    
    def integrate_brownian(self, X, n_steps, dt):
        step = 0
        X = np.array(X)
        X_t = [X]
        E_t = [self.potential(X)]
        while step < n_steps:
            X = X_t[step] + self.force(X_t[step]) * dt / self.gamma + self.random_force(dt = dt)
            X_t.append(X)
            E_t.append(self.potential(X))
            step += 1
        X_t = np.vstack(X_t)
        E_t = np.vstack(E_t)
        return X_t , E_t
