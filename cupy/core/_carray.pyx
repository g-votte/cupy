import os

from cupy.core cimport core
from cupy.cuda cimport function


include "_carray.pxi"


cdef class CArray(CPointer):

    def __init__(self, core.ndarray arr):
        cdef Py_ssize_t i
        cdef int ndim = arr._shape.size()
        self.val.data = <void*>arr.data.ptr
        self.val.size = arr.size
        for i in range(ndim):
            self.val.shape_and_strides[i] = arr._shape[i]
            self.val.shape_and_strides[i + ndim] = arr._strides[i]
        self.ptr = <void*>&self.val


cdef class CIndexer(CPointer):

    def __init__(self, Py_ssize_t size, tuple shape):
        self.val.size = size
        cdef Py_ssize_t i
        for i in range(len(shape)):
            self.val.shape_and_index[i] = shape[i]
        self.ptr = <void*>&self.val


cdef class Indexer:
    def __init__(self, tuple shape):
        cdef Py_ssize_t size = 1
        for s in shape:
            size *= s
        self.shape = shape
        self.size = size

    @property
    def ndim(self):
        return len(self.shape)

    cdef CPointer get_pointer(self):
        return CIndexer(self.size, self.shape)
