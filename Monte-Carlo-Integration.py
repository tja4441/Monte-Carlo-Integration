import numpy as np
import scipy as sp
from matplotlib import pyplot as pt
import test_functions as f


numPoints=1000

# uniform dist; [-10,10]
n10to10 = sp.stats.uniform(loc=-10, scale=20)
distOne = sp.stats.uniform(loc=1,scale=100)

def main():
    print("Starting program...")
    pt.plot(distOne.rvs(numPoints), distOne.pdf(distOne.rvs(numPoints)))
    pt.show()

def integrate(f, distrib, npts):
    x = distrib.rvs(npts)
    ps = distrib.pdf(x)
    f = f(x)
    mu = np.mean(f/ps)
    err = np.std(f/ps)/np.sqrt(npts)
    return mu,err

if __name__ == '__main__':
    main()