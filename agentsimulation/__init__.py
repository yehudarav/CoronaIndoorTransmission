import numpy
import scipy as sp
import scipy.interpolate

from unum import Unum
from unum.units import *
ml = Unum.unit('ml', 1e-3*L)

SUSCEPTIBLE = "Susceptible"
EXPOSED = "Exposed"
INFECTED = "Infection"
RECOVERED = "Recovered"

def log_interp1d(xx, yy, kind='linear'):

    logy = numpy.log10(yy)
    lin_interp = sp.interpolate.interp1d(xx, logy, kind=kind)
    log_interp = lambda zz: numpy.power(10.0, lin_interp(zz))
    return log_interp