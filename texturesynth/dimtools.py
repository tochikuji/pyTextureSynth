import numpy
import functools


def dimension_estimator(dimfun):
    @functools.wraps(dimfun)
    def inner(*args, **option):
        flatten = option.pop('flatten', False)
        dims = dimfun(*args, **option)

        if flatten:
            return numpy.prod(dims)
        else:
            return dims

    return inner


def dim_pss(nsc, nor, na):
    args = (nsc, nor, na)
    shapes = shape_pss(*args, flatten=True)
    return numpy.sum(shapes, dtype=int)


def shape_pss(nsc, nor, na, flatten=False):
    args = (nsc, nor, na)

    return (
        dim_pixelStats(*args, flatten=flatten),
        dim_pixelLPStats(*args, flatten=flatten),
        dim_autoCorrMag(*args, flatten=flatten),
        dim_autoCorrReal(*args, flatten=flatten),
        dim_magMeans(*args, flatten=flatten),
        dim_cousinMagCorr(*args, flatten=flatten),
        dim_parentMagCorr(*args, flatten=flatten),
        dim_cousinRealCorr(*args, flatten=flatten),
        dim_parentRealCorr(*args, flatten=flatten),
        dim_varianceHPR(*args, flatten=flatten)
    )


@dimension_estimator
def dim_pixelStats(_, __, ___):
    return (1, 6)


@dimension_estimator
def dim_pixelLPStats(nsc, _, __):
    return (2, nsc + 1)


@dimension_estimator
def dim_autoCorrReal(nsc, _, na):
    return (na, na, nsc + 1)


@dimension_estimator
def dim_autoCorrMag(nsc, nor, na):
    return (na, na, nor, nsc)


@dimension_estimator
def dim_magMeans(nsc, nor, _):
    return (nor * nsc + 2, 1)


@dimension_estimator
def dim_cousinMagCorr(nsc, nor, _):
    return (nor, nor, nsc + 1)


@dimension_estimator
def dim_parentMagCorr(nsc, nor, _):
    return (nor, nor, nsc)


@dimension_estimator
def dim_cousinRealCorr(nsc, nor, _):
    return (2 * nor, 2 * nor, nsc + 1)


@dimension_estimator
def dim_parentRealCorr(nsc, nor, _):
    return (2 * nor, 2 * nor, nsc)


@dimension_estimator
def dim_varianceHPR(_, __, ___):
    return tuple()


def estimate_hyperparam(dim, **hint):
    if 'nsc' in hint:
        nsc_iter = [hint['nsc']]
    else:
        nsc_iter = range(1, 5)

    if 'nor' in hint:
        nor_iter = [hint['nor']]
    else:
        nor_iter = range(1, 5)

    if 'na' in hint:
        na_iter = [hint['na']]
    else:
        na_iter = range(1, 50, 2)

    hpdict = dict()
    for nsc in nsc_iter:
        for nor in nor_iter:
            for na in na_iter:
                hpdict[(nsc, nor, na)] = int(dim_pss(nsc, nor, na))

    # pick duplicated items in hpdict
    dimset = set()
    duplicated_dims = set()

    for d in hpdict.values():
        if d in dimset:
            duplicated_dims.add(d)
        else:
            dimset.add(d)

    inverse_hpdict = {hpdim: hp for hp, hpdim in hpdict.items()}

    if dim in inverse_hpdict:
        estimation = inverse_hpdict[dim]
    else:
        raise LookupError('Cannot estimate hyperparameter from PSS '
                          'dimensionality. Dimensionality might be invalid '
                          'or came from too large hyperparameter.')

    if dim in duplicated_dims:
        raise ValueError('Cannot identify the hyperparameter of the PSS. '
                         '{} could have {}-dimensionality, respectively. '
                         'If you can, try to give hints of hyperparameter.'
                         .format(', '.join(
                             [str(shape) for shape, dupdim in hpdict.items()
                              if dupdim == dim]), dim))

    else:
        return estimation
