import numpy as np
import scipy.stats

distrib0 = scipy.stats.truncnorm(-3,3,loc=0,scale=1)
distrib1 = scipy.stats.uniform(scale=1)
numPoints=1000

def main():
    print(integrate_me(f1, scipy.stats.uniform(scale=1), numPoints))
    print(integrate_me(Pl0, distrib1, numPoints))

def f1(x):
    return 1

def Pl0(x):
    return (scipy.special.eval_legendre(0,x))**2

def integrate_me(f, distrib, npts):
    x = distrib.rvs(npts)
    ps = distrib.pdf(x)
    f = f(x)
    mu = np.mean(f/ps)
    err = np.std(f/ps)/np.sqrt(npts)
    return mu,err

if __name__ == '__main__':
    main()