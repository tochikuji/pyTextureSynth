from warnings import warn

import numpy
import scipy

import pyrtools as pyr
from .util import pyrBand, pyrBandIndices, expand, shift, vectify, pyrLow
from .fakesfpyr import FakeSFPyr
from .psstat import PSStat


class Synthesizer(object):
    """
    The texture synthesizer class using Simoncelli and Portilla's algorithm.
    """

    def __init__(self, pss, im0=None):
        self.pss = pss
        self.statg0 = self.pss.pixelStats
        self.mean0 = self.statg0[0]
        self.var0 = self.statg[1]
        self.skew0 = self.statg[2]
        self.kurt0 = self.statg[3]
        self.mn0 = self.statg[4]
        self.mx0 = self.statg[5]
        self.statsLPim = self.pss.pixelLPStats
        self.skew0p = self.statsLPim[:, 0]
        self.kurt0p = self.statsLPim[:, 1]
        self.vHPR0 = self.varianceHPR
        self.acr0 = self.autoCorrReal
        self.ace0 = self.pss.autoCorrMag
        self.magMeans0 = self.pss.magMeans
        self.C0 = self.pss.cousinMagCorr
        self.Cx0 = self.pss.parentMagCorr
        self.Crx0 = self.pss.parentRealCorr

    def synthesize(self, n_iter=1):
        pass

    def next(self):
        pass
