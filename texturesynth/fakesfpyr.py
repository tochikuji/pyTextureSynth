import pyrtools as pyr
import numpy


class FakeSFPyr(pyr.SFpyr):
    def __init__(self, pyr, pind):
        self.pyr = list()
        self.pyrSize = pind

        # decompose pyr vector into each bands
        start = 0
        for shape in pind:
            ind = numpy.prod(shape)
            self.pyr.append(pyr[start:start + ind].reshape(*shape))
            start += ind
