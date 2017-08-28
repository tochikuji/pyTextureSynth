import numpy


fkeys = ['pixelStats',
         'pixelLPStats',
         'autoCorrReal',
         'autoCorrMag',
         'magMeans',
         'cousinMagCorr',
         'parentMagCorr',
         'cousinRealCorr',
         'parentRealCorr',
         'varianceHPR']


class PSStat(object):

    def __init__(self, *param, **dic):

        # param is empty
        if len(param) == 0 and len(dic) == 0:
            for name in fkeys:
                setattr(self, name, None)

        elif len(param) != 0:
            if len(param) != len(fkeys):
                raise ValueError("length of unnamed param does not match"
                                 " for construct the PSS")

            for v, name in zip(param, fkeys):
                setattr(self, name, v)

        else:
            if set(fkeys) not in set(dic.keys):
                raise KeyError("named parameter must contain all of fkeys."
                               "refer to .fkeys")

            for name in fkeys:
                setattr(self, name, dic[name])

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError("key must be string.")
        if key not in fkeys:
            raise KeyError("Cannot access key {} without .fkeys".format(key))

        return getattr(self, key)

    @property
    def data(self):
        return [numpy.asarray(field) for field in [
            self.__getitem__(key) for key in fkeys
        ]]

    @property
    def vec(self):
        return numpy.hstack([
            x.reshape(-1) for x in self.data
        ])

    def dims(self, flatten=False):
        dimensions = [x.shape for x in self.data]

        if flatten:
            return [numpy.prod(d) for d in dimensions]
        else:
            return dimensions

    def dic(self, flatten=False):
        if flatten:
            return {key: self.__getitem__(key).reshape(-1) for key in fkeys}
        else:
            return {key: self.__getitem__(key) for key in fkeys}
