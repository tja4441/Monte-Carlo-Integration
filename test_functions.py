#############################################
# Contains test functions for use in the
# Monte-Carlo-Integration file
#############################################
import scipy as sp

def one(x): # f(x)=1
    return 0*x+1

def Pl0(x): # f(x)=P_0(x)^2
    return (sp.special.eval_legendre(0,x))**2

def Pl1(x): # f(x)=P_1(x)^2
    return (sp.special.eval_legendre(1,x))**2

def Pl2(x): # f(x)=P_2(x)^2
    return (sp.special.eval_legendre(2,x))**2

def Pl3(x): # f(x)=P_3(x)^2
    return (sp.special.eval_legendre(3,x))**2

def Pl4(x): # f(x)=P_4(x)^2
    return (sp.special.eval_legendre(4,x))**2

def Pl5(x): # f(x)=P_5(x)^2
    return (sp.special.eval_legendre(5,x))**2

def snPDF(x): # f(x)=PDF of std norm var
    return (sp.stats.norm(loc=0,scale=1).pdf(x))

# p_1(x)=PDF at x of norm; mean -3, stddev 1
# p_2(x)=PDF at x of norm; mean 3, stddev 3
def weightedsum(x): # f(x)=0.7*p_1(x)+0.3*p_2(x)
    return (0.7*sp.stats.norm(loc=-3,scale=1).pdf(x)+
            0.3*sp.stats.norm(loc=3,scale=3).pdf(x))