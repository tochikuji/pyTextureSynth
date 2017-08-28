from warnings import warn

import numpy
import scipy

import pyrtools as pyr
from .util import pyrBand, pyrBandIndices, expand, shift, vectify, pyrLow
from .fakesfpyr import FakeSFPyr
from .psstat import PSStat


def analyze(img, Nsc=4, Nor=4, Na=7, preview=False):
    """
    Because I have no ESP skills, this is *just* my guess.

    Nsc : Number of scales
    Nor : Number of orders
    Na  : Number of ???? (neighbor? but has no 'a')
    """

    img = numpy.asarray(img, dtype=numpy.float64)
    if len(img.shape) != 2:
        raise ValueError("image must be 2-dimensional grayscaled, "
                         "that has (height, width) shape")

    Nsc = Nsc
    Nor = Nor
    Na = Na

    if Na % 2 == 0:
        raise ValueError("Neighbor parameter Na could not be even number")

    height, width = img.shape

    nth = numpy.log2(min(height, width) / Na)
    if nth < Nsc:
        warn("Na will be cut off for levels above {}".format(int(nth) + 1))

    la = int(numpy.floor((Na - 1) / 2))

    mn0 = numpy.min(img)
    mx0 = numpy.max(img)

    mean0 = numpy.mean(img)
    var0 = numpy.var(img)
    skew0 = scipy.stats.skew(img, axis=None)
    kurt0 = scipy.stats.kurtosis(img, fisher=False, axis=None)

    statg0 = numpy.asarray((mean0, var0, skew0, kurt0, mn0, mx0))

    # adopt a little bit of noise to the original image.
    img += (mx0 - mn0) / 1000 * numpy.random.randn(height, width)

    sfpyr = pyr.SCFpyr(img, Nsc, Nor - 1)
    pyr0 = sfpyr.pyr
    pind0 = sfpyr.pyrSize

    # subtract mean of lowband [81-92]
    nband = len(pind0)
    for band in pyr0:
        band = numpy.real(band) - numpy.mean(numpy.real(band))

    rpyr0 = numpy.real(pyr0)
    apyr0 = numpy.abs(pyr0)

    if preview:
        pyr.showIm(img, "Original")

    # Subtract mean of magnitude
    magmean0 = numpy.zeros((nband, 1))

    for n in range(nband):
        magmean0[n] = numpy.mean(apyr0[n])
        apyr0[n] = apyr0[n] - magmean0[n]

    # Compute central autoCoor of lowband [102-125]
    acr = numpy.zeros((Na, Na, Nsc + 1))
    ch = pyrBand(pyr0, pind0, nband - 1)

    msfpyr = pyr.SFpyr(numpy.real(ch), 0, 0)
    mpyr = msfpyr.pyr
    mpind = msfpyr.pyrSize

    im = pyrBand(mpyr, mpind, 1)

    Nly, Nlx = ch.shape
    Sch = min(Nly, Nlx)

    le = min(int(Sch / 2 - 1), int(la))

    cy = int(Nly / 2 + 1)
    cx = int(Nlx / 2 + 1)

    ac = numpy.fft.fftshift(numpy.real(numpy.fft.ifft2(
        numpy.abs(numpy.fft.fft2(im)) ** 2))) / numpy.prod(ch.shape)
    ac = ac[cy - le:cy + le, cx - le:cx + le]
    acr[la - le:la + le, la - le:la + le, Nsc] = ac

    skew0p = numpy.zeros((Nsc + 1, 1))
    kurt0p = numpy.zeros((Nsc + 1, 1))
    vari = ac[le, le]

    if vari / var0 > 1e-6:
        skew0p[Nsc] = numpy.mean(im ** 3) / (vari ** 1.5)
        kurt0p[Nsc] = numpy.mean(im ** 4) / (vari ** 2)
    else:
        skew0p[Nsc] = 0
        kurt0p[Nsc] = 3

    # Compute central autoCorr of each magband, and the autocorr of the
    # combined band [127-142]
    ace = numpy.zeros((Na, Na, Nsc, Nor))

    for nsc in range(Nsc)[::-1]:
        for nor in range(Nor):
            n = (nsc - 1) * Nor + nor
            ch = pyrBand(apyr0, pind0, nband - 1)

            Nly, Nlx = ch.shape
            Sch = min(Nly, Nlx)

            le = min(int(Sch / 2 - 1), int(la))
            cx = int(Nlx / 2 + 1)
            cy = int(Nly / 2 + 1)

            ac = numpy.fft.fftshift(numpy.real(numpy.fft.ifft2(
                numpy.abs(numpy.fft.fft2(ch)) ** 2))) / numpy.prod(ch.shape)
            ac = ac[cy - le:cy + le, cx - le:cx + le]
            ace[la - le:la + le, la - le:la + le, nsc, nor] = ac

        # combine orientation bands [144-149]
        bandNums = numpy.arange(0, Nor) + nsc * Nor + 1

        # Make fake pyramid [150-168]
        fakePind = numpy.vstack([
            pind0[bandNums[0]],
            *[pind0[i] for i in range(bandNums[0],
                                      bandNums[0] + len(bandNums) + 1)]
        ])

        zs = numpy.zeros((numpy.prod(fakePind[0]), 1))
        zdata = numpy.vstack([rpyr0[i] for i in bandNums]).reshape(-1, 1)
        zd = numpy.zeros((numpy.prod(fakePind[len(fakePind) - 1]), 1))

        fakePyr_pyr = numpy.vstack([
            zs, zdata, zd
        ])

        fakePyr = FakeSFPyr(fakePyr_pyr, fakePind)
        ch = fakePyr.reconPyr([0])

        im = numpy.real(expand(im, 2)) / 4
        im = im + ch
        ac = numpy.fft.fftshift(numpy.real(numpy.fft.ifft2(numpy.abs(
            numpy.fft.fft2(im)) ** 2))) / numpy.prod(ch.shape)

        ac = ac[cy - le:cy + le, cx - le:cx + le]
        acr[la - le:la + le, la - le:la + le, nsc] = ac
        vari = ac[le, le]

        if vari / var0 > 1e-6:
            skew0p[nsc] = numpy.mean(im ** 3) / (vari ** 1.5)
            kurt0p[nsc] = numpy.mean(im ** 4) / (vari ** 2)
        else:
            skew0p[nsc] = 0
            kurt0p[nsc] = 3

    # Compute the cross-correlation of the coeff. magnitudes
    # [170-228]
    C0 = numpy.zeros((Nor, Nor, Nsc + 1))
    Cx0 = numpy.zeros((Nor, Nor, Nsc))

    Cr0 = numpy.zeros((Nor * 2, Nor * 2, Nsc + 1))
    Crx0 = numpy.zeros((Nor * 2, Nor * 2, Nsc))

    # TODO: fix cousinInd
    apyr0 = numpy.hstack([p.reshape(-1) for p in apyr0])
    pyr0_ = numpy.hstack([p.reshape(-1) for p in pyr0])

    for nsc in range(Nsc):
        firstBnum = nsc * Nor + 1  # ??
        cousinSz = numpy.prod(pind0[firstBnum])
        ind = pyrBandIndices(pind0, firstBnum)
        # TODO: fix cousinInd
        cousinInd = ind[0] + numpy.arange(0, Nor * cousinSz,
                                          dtype=numpy.int32)

        if nsc < Nsc - 1:
            parents = numpy.zeros((cousinSz, Nor))
            rparents = numpy.zeros((cousinSz, Nor * 2))

            for nor in range(Nor):
                nband = (nsc + 1) * Nor + nor + 1

                tmp = expand(pyrBand(pyr0, pind0, nband), 2) / 4
                rtmp = numpy.real(tmp)
                itmp = numpy.imag(tmp)

                tmp = numpy.sqrt(rtmp ** 2 + itmp ** 2) * \
                    numpy.exp(2j * numpy.arctan2(rtmp, itmp))
                rparents[:, nor] = numpy.real(tmp).reshape(-1)
                rparents[:, nor + Nor] = numpy.imag(tmp).reshape(-1)

                tmp = abs(tmp)
                parents[:, nor] = (tmp - numpy.mean(tmp)).reshape(-1)

        else:
            tmp = numpy.real(expand(pyrLow(rpyr0, pind0), 2)) / 4
            rparents = numpy.hstack([
                vectify(tmp),
                vectify(shift(tmp, [0, 1])),
                vectify(shift(tmp, [0, -1])),
                vectify(shift(tmp, [1, 0])),
                vectify(shift(tmp, [-1, 0]))
            ])

            parents = []

        # TODO: fix cousinInd
        cousins = apyr0[cousinInd].reshape(cousinSz, Nor)
        nc = numpy.asarray(cousins).shape[-1]
        np = numpy.asarray(parents).shape[-1]

        C0[0:nc, 0:nc, nsc] = numpy.dot(cousins.T, cousins) / cousinSz

        if np > 0:
            Cx0[0:nc, 0:np, nsc] = numpy.dot(cousins.T, parents) / cousinSz

            if nsc == Nsc - 1:
                C0[0:np, 0:np, Nsc] = numpy.dot(parents.T, parents) / \
                    cousinSz * 4

        # TODO: fix cousinInd
        cousins = numpy.real(pyr0_[cousinInd]).reshape(cousinSz, Nor)
        nrc = cousins.shape[1]
        nrp = rparents.shape[1]

        Cr0[0:nrc, 0:nrc, nsc] = numpy.dot(cousins.T, cousins) / cousinSz

        if nrp > 0:
            Crx0[0:nrc, 0:nrp, nsc] = numpy.dot(cousins.T, rparents) / \
                cousinSz
            if nsc == Nsc - 1:
                Cr0[0:nrp, 0:nrp, Nsc] = numpy.dot(rparents.T, rparents) /\
                    cousinSz * 4

    # Calculate the mean, range and var. of te LF and HF residual
    # [230-235]
    channel = pyr0[0]
    vHPR0 = numpy.mean(channel ** 2)

    statsLPim = numpy.array([skew0p, kurt0p])

    ret = PSStat(
        statg0,
        statsLPim,
        acr,
        ace,
        magmean0,
        C0,
        Cx0,
        Cr0,
        Crx0,
        vHPR0
    )

    return ret
