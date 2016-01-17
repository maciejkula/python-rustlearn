import os

import cffi

import numpy as np


def _build_bindings():

    ffi = cffi.FFI()
    ffi.cdef("""
    struct SGDModel;
    struct SGDModel *fit_sgdclassifier(float *X_ptr, unsigned long X_rows, unsigned long X_cols,
                                float *y_ptr, unsigned long y_rows, unsigned long y_cols);
    float* predict_sgdclassifier(float *X_ptr, unsigned long X_rows, unsigned long X_cols,
                                 struct SGDModel *model);
    void free_sgdclassifier(struct SGDModel *model);
    """)

    path  = os.path.join(os.path.dirname(__file__),
                         'rustlearn-bindings/target/release/librustlearn.so')
    lib = ffi.dlopen(path)

    return ffi, lib


ffi, lib = _build_bindings()


def _as_float(array):
    """
    Cast a np.float32 array to a float*.
    """

    return ffi.cast('float*', array.ctypes.data)


def _as_usize(num):
    """
    Cast num to something like a rust usize.
    """

    return ffi.cast('unsigned long', num)


def _as_float_ndarray(ptr, size):
    """
    Turn a float* to a numpy array.
    """

    return np.core.multiarray.int_asbuffer(ptr, size * np.float32.itemsize)


class SGDClassifier(object):

    def __init__(self):
        self.model = None

    def fit(self, X, y):

        self.model = lib.fit_sgdclassifier(_as_float(X),
                                           _as_usize(X.shape[0]),
                                           _as_usize(X.shape[1]),
                                           _as_float(y),
                                           _as_usize(len(y)),
                                           _as_usize(1))

    def predict(self, X):

        if self.model is None:
            raise Exception('Call fit before calling predict')

        predictions_ptr = lib.predict_sgdclassifier(
            _as_float(X), _as_usize(X.shape[0]), _as_usize(X.shape[1]),
            ffi.cast('struct SGDModel*', self.model)
        )

        predictions = np.frombuffer(ffi.buffer(predictions_ptr, len(X)
                                               * ffi.sizeof('float')),
                                    dtype=np.float32)

        return predictions

    def __del__(self):

        if self.model is not None:
            lib.free_sgdclassifier(self.model)
            self.model = None
