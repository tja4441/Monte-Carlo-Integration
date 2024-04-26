import numpy as np
import scipy as sp
from matplotlib import pyplot as pt
import test_functions as f

class MyTwoDUniform(object):
    def __init__(self, bounds=None):
        self.bounds =np.array(bounds)
    def rvs(self,npts):
        my_out = np.empty( (len(self.bounds),npts))
        for dim in np.arange(len(self.bounds)):
            my_out[dim] = np.random.uniform(low=self.bounds[dim][0], high=self.bounds[dim][1], size=npts)
        return my_out.T
    def pdf(self,x):
        V = np.prod([self.bounds[:,1]- self.bounds[:,0]])
        return np.ones(x.shape[0])/V
    
numPoints=1000000000

# uniform dist; [-10,10]
un10to10 = sp.stats.uniform(loc=-10, scale=20)
# uniform dist; [-1,1]
un1to1 = sp.stats.uniform(loc=-1, scale=2)

snpdfn10to10 = sp.stats.truncnorm(a=-10, b=10)

snpdfn1to1 = sp.stats.truncnorm(a=-1, b=1)

n5to5_2d = MyTwoDUniform(bounds=[[5,-5],[-5,5]])

mu = np.array([1,-0.5])
cov = np.array([[ 1. , -0.1 ], [-0.1 , 0.05]])

def main():
    printegrate(lambda x: ln_f(*(x.T)), n5to5_2d, numPoints)

def ln_f(x1, x2):
    minus_lnL = np.array(np.power((1.-x1), 2) + 100.* np.power((x2-x1**2),2),dtype=float)
    return - minus_lnL

def printegrate(f, distrib, npts):
    integrand = integrate(f, distrib, npts)
    print("mu = "+ str(integrand[0]))
    print("error = "+ str(integrand[1]))
    print("")

def integrate(f, distrib, npts):
    x = distrib.rvs(npts)
    ps = distrib.pdf(x)
    f = f(x)
    mu = np.mean(f/ps)
    err = np.std(f/ps)/np.sqrt(npts)
    return mu,err

def plot(f, dist, npts):
    pts = sorted(dist.rvs(npts))
    pt.plot(pts, f(pts))
    pt.show()

def scatter(f, dist, npts):
    pts = dist.rvs(npts)
    pt.scatter(pts, f(pts))
    pt.show()

if __name__ == '__main__':
    main()