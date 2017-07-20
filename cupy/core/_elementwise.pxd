from cupy.core.kernel_core cimport ParameterList


cdef class _BaseElementwiseKernel:
    cdef:
        readonly tuple in_params
        readonly tuple out_params
        readonly tuple inout_params
        readonly int nin
        readonly int nout
        readonly int nargs
        readonly tuple params
        readonly str name
        readonly bint reduce_dims
        readonly dict kernel_cache

    cpdef call(self, args, kwargs)
    cpdef create_call_context(self, args, int size, stream, kwargs)


cdef class ElementwiseKernel(_BaseElementwiseKernel):

    cdef:
        readonly str operation
        readonly str preamble

    cpdef create_call_context(self, args, int size, stream, kwargs)


cpdef object create_ufunc(name, ops, routine=*, preamble=*, doc=*)
