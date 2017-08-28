import numpy
from PIL import Image, ImageChops


def pyrBandIndices(pind, band):
    """
    pind : (int * int) list
    band : int

    returns index range of specified band.
    e.g. if pind = ([3, 3], [2, 2], [1, 1]), band = 1
         then range(9, 13)  (=range(3*3, 3*3+2*2))

    This method was frequently used in original matlab code,
    due to the pyr is vectified.
    pyPyrTools provides the list of 2-D pyramid subbands,
    which is accessible with pyr[band] returns np.array.
    So this method would be useless.
    """

    ind = 0
    for i in range(band):
        ind += numpy.prod(pind[i])

    return range(ind, ind + numpy.prod(pind[band]))


def pyrBand(pyr, _, band):
    return pyr[band]


def expand(img, factor):
    """
    expand image spatially.
    img : image (might be complex)
    factor : expand factor
    """

    height, width = numpy.asarray(img).shape
    mh = factor * height
    mw = factor * width

    Te = numpy.zeros((mh, mw), dtype=numpy.complex)
    # source image in Fourier domain
    fimg = (factor ** 2) * numpy.fft.fftshift(numpy.fft.fft2(img))

    y1 = int(mh / 2 - mh / (2 * factor))
    y2 = int(mh / 2 + mh / (2 * factor))
    x1 = int(mw / 2 - mw / (2 * factor))
    x2 = int(mw / 2 + mw / (2 * factor))

    Te[y1:y2, x1:x2] = fimg[0:int(mh / factor), 0:int(mw / factor)]
    Te[y1 - 1, x1:x2] = fimg[0, 0:int(mw / factor)] / 2
    Te[y2 + 1, x1:x2] = numpy.conjugate(fimg[0, 0:int(mw / factor)][::-1] / 2)
    Te[y1:y2, x1 - 1] = fimg[0:int(mh / factor), 0] / 2
    Te[y1:y2, x2 + 1] = numpy.conjugate(fimg[0:int(mh / factor), 0][::-1]) / 2

    esq = fimg[0, 0] / 4

    Te[y1 - 1, x1 - 1] = esq
    Te[y1 - 1, x2 + 1] = esq
    Te[y2 + 1, x1 - 1] = esq
    Te[y2 + 1, x2 + 1] = esq

    Te = numpy.fft.fftshift(Te)
    dst = numpy.fft.ifft2(Te)

    if (numpy.imag(img) <= 1e-10).all():
        dst = numpy.real(dst)

    return dst


def pyrLow(pyr, _):
    return pyr[-1]


def shift(mtx, offset):
    """
    make image offsets.
    using pillow.ImageChops
    """

    img = Image.fromarray(mtx)
    ret = ImageChops.offset(img, offset[1], offset[0])

    return numpy.asarray(ret)


def vectify(mtx):
    return numpy.asarray(mtx).reshape(-1, 1)
