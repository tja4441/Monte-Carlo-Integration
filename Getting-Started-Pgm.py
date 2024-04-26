import numpy as np
import scipy as sp
from matplotlib import pyplot as pt
import test_functions as f


numPoints=1000

# uniform dist; [-10,10]
un10to10 = sp.stats.uniform(loc=-10, scale=20)
# uniform dist; [-1,1]
un1to1 = sp.stats.uniform(loc=-1, scale=2)

snpdfn10to10 = sp.stats.truncnorm(a=-10, b=10)

snpdfn1to1 = sp.stats.truncnorm(a=-1, b=1)

def main():
    print("Starting program...")
    ########## Integral estimates 0: Uniform, 1d ##########
    print("\n---------------- Integral estimates 0: Uniform, 1d ----------------\n")
    #Q1
    printegrate(f.one,un10to10,numPoints)
    #Q2
    printegrate(f.Pl0, un1to1, numPoints)
    printegrate(f.Pl1, un1to1, numPoints)
    printegrate(f.Pl2, un1to1, numPoints)
    printegrate(f.Pl3, un1to1, numPoints)
    printegrate(f.Pl4, un1to1, numPoints)
    printegrate(f.Pl5, un1to1, numPoints)
    #Q3
    printegrate(f.snPDF, un10to10, numPoints)
    #Q4
    printegrate(f.weightedsum, un10to10, numPoints)

    ########## Integral estimates 1: Gaussian, 1d ##########
    print("\n---------------- Integral estimates 1: Gaussian, 1d ----------------\n")
    #Q1
    printegrate(f.one,snpdfn10to10,numPoints)
    #Q3
    printegrate(f.snPDF, snpdfn10to10, numPoints)
    #Q4
    printegrate(f.weightedsum, snpdfn10to10, numPoints)

def integrate(f, distrib, npts):
    x = distrib.rvs(npts)
    ps = distrib.pdf(x)
    f = f(x)
    mu = np.mean(f/ps)
    err = np.std(f/ps)/np.sqrt(npts)
    return mu,err

def printegrate(f, distrib, npts):
    print("Integrating",f.__name__,"from",str(distrib.ppf(0)),"to",
            str(distrib.ppf(1)),"using",str(npts),"points...")
    integrand = integrate(f, distrib, npts)
    print("mu = "+ str(integrand[0]))
    print("error = "+ str(integrand[1]))
    print("")
    

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