from cupy.cuda.function cimport CPointer


include "_carray.pxi"


cdef struct _CArray:
    void* data
    Py_ssize_t size
    Py_ssize_t shape_and_strides[MAX_NDIM * 2]


cdef class CArray(CPointer):

    cdef:
        _CArray val


cdef struct _CIndexer:
    Py_ssize_t size
    Py_ssize_t shape_and_index[MAX_NDIM * 2]


cdef class CIndexer(CPointer):
    cdef:
        _CIndexer val


cdef class Indexer:
    cdef:
         readonly Py_ssize_t size
         readonly tuple shape

    cdef CPointer get_pointer(self)
