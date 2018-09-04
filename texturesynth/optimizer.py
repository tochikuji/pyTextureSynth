import numpy
from warnings import warn

from util import vectify


def modacor22(X, Cy, p=1):
    Ny, Nx = X.shape
    Nc = len(Cy)

    if 2 * Nc - 1 > Nx:
        warn('Autocorrelation neighborhood too large for image. '
             'This would be reduced.')

        Nc = int(2 * numpy.floor(Nx / 4) - 1)
        first = int(len(Cy) - Nc / 2)
        Cy = Cy[first:first + Nc, first:first + Nc]

    Xf = numpy.fft.fft2(X)
    Xf2 = numpy.abs(Xf) ** 2
    isreal = int(numpy.all(numpy.isreal(X)))
    Cx = numpy.fft.fftshift(numpy.real(numpy.fft.ifft2(Xf2))) / (2 - isreal)
    Cy = Cy * numpy.prod(X.shape)

    cy = int(Ny / 2 + 1)
    cx = int(Nx / 2 + 1)
    Lc = (Nc - 1) / 2

    Cy0 = Cy
    Cy = p * Cy + (1 - p) * Cx[cy - Lc:cy + Lc, cx - Lc:cx + Lc]

    snrV = 10 * numpy.log10(numpy.sum(Cy0 ** 2)) / \
        numpy.sum((Cy0 - Cx[cy - Lc:cy + Lc, cx - Lc:cx + Lc]) ** 2)

    Cx = Cx[cy - 2 * Lc:cy + 2 * Lc, cx - 2 * Lc:cx + 2 * Lc]

    Ncx = 4 * Lc + 1
    M = int((Nc ** 2 + 1) / 2)
    Tcx = numpy.zeros(M)

    for i in range(Lc, 2 * Lc):
        for j in range(Lc, 3 * Lc):
            nm = (i - Lc - 1) * (2 * Lc + 1) + j - Lc
            ccx = Cx[i - Lc:i + Lc, j - Lc:j + Lc]
            ccxi = ccx[::-1, ::-1]
            ccx = ccx + ccxi

            ccx[Lc, Lc] = ccx[Lc, Lc] / 2
            ccx = vectify(ccx.T)
            Tcx[nm, :] = ccx[0:M].T

    i = 2 * Lc

    for j in range(Lc, 2 * Lc):
        nm = (i - Lc - 1) * (2 * Lc + 1) + j - Lc
        ccx = Cx[i - Lc:i + Lc, j - Lc:j + Lc]
        ccxi = ccx[::-1, ::-1]
        ccx = ccx + ccxi
        ccx[Lc, Lc] = ccx[Lc, Lc] / 2
        ccx = vectify(ccx.T)
        Tcx[nm, :] = ccx[0:M].T

    Cy1 = vectify(Cy.T)
    Cy1 = Cy1[0:M]

    Ch1 = numpy.linalg.solve(Tcx, Cy1)

    # concat folding back without tail element
    Ch1 = numpy.hstack([Ch1, Ch1[0:-1][::-1]])

    Ch = Ch1.reshape(Nc, Nc)

    aux = numpy.zeros(Ny, Nx)
    aux[cy - Lc:cy + Lc, cx - Lc:cx + Lc] = Ch
    Ch = numpy.fft.fftshift(aux)
    Chf = numpy.real(numpy.fft.fft2(Ch))
    Yf = Xf * numpy.sqrt(numpy.abs(Chf))
    Y = numpy.fft.ifft2(Yf)

    return Y, snrV, Chf


def modskew(ch, sk, p=1):
    N = numpy.prod(ch.shape)
    me = numpy.mean(ch)
    ch = ch - me

    # the index correspond to the order of a power
    # we must pay attention to such indexing
    m = numpy.zeros(6, ch.shape[0], ch.shape[1])
    for n in range(7):
        m[n] = numpy.mean(ch ** n)

    sd = sqrt(m[2])
    s = m[3] / (sd ** 3)
    snrk = snr[sk, sk - s]
    sk = s * (1 - p) + sk * p

    # [42-53]
    A = m[6] - 3 * sd * s * m[5] + 3 * (sd ** 2) * (s ** 2-1) * m[4] + \
        (sd ** 6) * (2 + 3 * s ** 2 - s ** 4)
    B = 3 * (m[5] - 2 * sd * s * m[4] + sd ** 5 * s ** 3)
    C = 3 * (m[4] - sd ** 4 * (1 + s ** 2))
    D = s * sd ** 3

    a = numpy.zeros(7)
    a[6] = A ** 2
    a[5] = 2 * A * B
    a[4] = B ** 2 + 2 * A * C
    a[3] = 2 * (A * D + B * C)
    a[2] = C ** 2 + 2 * B * D
    a[1] = 2 * C * D
    a[0] = D ** 2

    # [57-64]
    A2 = sd ** 2
    B2 = m[4] - (1 + s ** 2) * sd ** 4

    b = numpy.zeros(7)
    b[6] = B2 ** 3
    b[4] = 3 * A2 * B2 ** 2
    b[2] = 3 * A2 ** 2 * B2
    b[0] = A2 ** 3

    # [86-96]
    d = numpy.zeros(8)
    d[7] = B * b[6]
    d[6] = 2 * C * b[6] - A * b[4]
    d[5] = 3 * D * b[6]
    d[4] = C * b[4] - 2 * A * b[2]
    d[3] = 2 * D * b[4] - B * b[2]
    d[2] = -3 * A * b[0]
    d[1] = D * b[2] - 2 * B * b[0]
    d[0] = -C * b[0]

    d = reversed(d)
    mMlambda = numpy.roots(d)

    # [98-114]
    tg = numpy.imag(mMlambda) / numpy.real(mMlambda)
    mMlambda = mMlambda
