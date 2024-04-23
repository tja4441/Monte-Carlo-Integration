import numpy as np
import scipy as sp
from matplotlib import pyplot as pt
import test_functions as f


numPoints=10000

# uniform dist; [-10,10]
un10to10 = sp.stats.uniform(loc=-10, scale=20)
un1to1 = sp.stats.uniform(loc=-1,scale=2)

def main():
    print("Starting program...")
    printegrate(f.Pl3,un1to1,numPoints)
    scatter(f.Pl3,un1to1,numPoints)

def integrate(f, distrib, npts):
    x = distrib.rvs(npts)
    ps = distrib.pdf(x)
    f = f(x)
    mu = np.mean(f/ps)
    err = np.std(f/ps)/np.sqrt(npts)
    return mu,err

def printegrate(f, distrib, npts):
    print("Integrating "+f.__name__+" from "+str(distrib.ppf(0))+" to "+str(distrib.ppf(1)))
    integrand = integrate(f, distrib, npts)
    print("mu = "+ str(integrand[0]))
    print("error = "+ str(integrand[1]))
    

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