from warnings import warn

import numpy
import scipy

import pyrtools
from .util import pyrBand, pyrBandIndices, expand, shift, vectify, pyrLow
from .fakesfpyr import FakeSFPyr
from .psstat import PSStat
from .optimizer import modacor22, modskew, modkurt


class Synthesizer(object):
    """
    The texture synthesizer class using Simoncelli and Portilla's algorithm.
    """

    def __init__(self, pss, im0, constraints=None):
        self.pss = pss
        self.statg0 = self.pss.pixelStats
        self.mean0 = self.statg0[0]
        self.var0 = self.statg0[1]
        self.skew0 = self.statg0[2]
        self.kurt0 = self.statg0[3]
        self.mn0 = self.statg0[4]
        self.mx0 = self.statg0[5]
        self.statsLPim = self.pss.pixelLPStats
        self.skew0p = self.statsLPim[:, 0]
        self.kurt0p = self.statsLPim[:, 1]
        self.vHPR0 = self.pss.varianceHPR
        self.acr0 = self.pss.autoCorrReal
        self.ace0 = self.pss.autoCorrMag
        self.magMeans0 = self.pss.magMeans
        self.C0 = self.pss.cousinMagCorr
        self.Cx0 = self.pss.parentMagCorr
        self.Crx0 = self.pss.parentRealCorr

        self.Nsc = self.pss.Nsc
        self.Nor = self.pss.Nor
        self.Na = self.pss.Na

        self.la = int((self.Na - 1) / 2)

        # set init image directly
        if isinstance(im0, numpy.ndarray):
            self.height, self.width = im0.shape
            self.im = im0
        # size given
        elif len(im0) == 2:
            self.height, self.width = im0
            # initialize with Gaussian noise
            self.im = self.mean0 + \
                numpy.random.randn(*im0) * numpy.sqrt(self.var0)

        if constraints is not None:
            if len(constraints) != 4:
                raise ValueError('constraints must be 4-dimensional '
                                 'boolean vector')
            self.cmask = constraints

        else:
            self.cmask = numpy.ones(4)

        # validate conditions
        nth = numpy.log2(min(self.height, self.width) / self.Na)
        if nth < self.Nsc + 1:
            warn("Na will be cut off for levels above {}".format(nth))

        self.nq = 0
        self.p = 1

    def next(self):
        sfpyr = pyrtools.SCFpyr(self.im, self.Nsc, self.Nor - 1)
        pyr = sfpyr.pyr
        pind = sfpyr.pyrSize

        # subtract mean of lowband [152-]
        nband = len(pind)
        pyr[-1] = pyr[-1] - numpy.mean(pyr[-1])

        apyr = numpy.abs(pyr)

        # adjust autoCorr of lowband [160-181]
        ch = pyr[-1]
        Sch = min(ch.shape) / 2
        nz = numpy.count_nonzero(~numpy.isnan(self.acr0[:, :, self.Nsc]))
        lz = (numpy.sqrt(nz) - 1) / 2
        le = min(Sch / 2 - 1, lz)
        im = numpy.real(ch)

        mpyr_pyr = pyrtools.SFpyr(im, 0, 0)
        mpyr = mpyr_pyr.pyr
        mpind = mpyr_pyr.pyrSize

        im = mpyr[1]
        la = self.la
        vari = self.acr0[la:la, la:la, self.Nsc]

        # if subband constraints is available
        if self.cmask[1]:
            if vari / var0 > 1e-4:
                # TODO: store snr2
                self.im, _, __ = modacor22(
                    im, self.acr0[la - le:la + le, la - le:la + le, self.Nsc],
                    self.p
                )
            else:
                im = im * numpy.sqrt(vari / numpy.var(im))

            if numpy.var(numpy.imag(ch)) / numpy.var(numpy.real(var)) > 1e-6:
                im = numpy.real(im)

        if self.cmask[0]:
            if vari / var0 > 1e-4:
                # TODO: store snr7
                self.im, _ = modskew(self.im, self.skew0p[self.Nsc + 1], self.p)
                self.im, _ = modkurt(self.im, self.kurt0p[self.Nsc + 1], self.p)


        # subtract mean of magnitude [189-197]
        if self.cmask[2]:
            magMeans = numpy.zeros(nband, 1)
            # n here means nband due to the name overrapping to constant
            for n in range(nband):
                magMeans[n] = numpy.mean(apyr[n])
                apyr[n] = apyr[n] - magMeans[n]

        # Coarse-to-fine loop [200-]
        for nsc in range(Nsc)[::-1]:
            firstBnum = nsc * self.Nor + 1
            cousinSz = 1

            if self.cmask[2] or self.cmask[3]:
                if nsc < Nsc:
                    parents = numpy.zeros((cousinSz, Nor))
                    rparents = numpy.zeros((cousinSz, Nor * 2))

                    for nor in range(Nor):
                        nband = 0x111
                        tmp = expand(pyr[nband], 2) / 4.
                        rtmp = numpy.real(tmp)
                        itmp = numpy.imag(tmp)
                        tmp = numpy.sqrt(rtmp ** 2 + itmp ** 2) * numpy.exp(2j * numpy.arctan(rtmp, itmp))
                        rparents[:, nor] = vectify(numpy.real(tmp))
                        rparents[:, Nor + nor] = vectify(numpy.imag(tmp))

                        tmp = numpy.abs(tmp)
                        parents[:, nor] = vectify(tmp - numpy.mean(tmp))
                else:
                    rparents = numpy.array([])
                    parents = numpy.array([])

            if cmask[2]:


    def synthesize(self, n_iter=50):
        for i in range(n_iter):
            self.next()
            return self.im
