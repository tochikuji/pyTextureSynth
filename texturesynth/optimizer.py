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
