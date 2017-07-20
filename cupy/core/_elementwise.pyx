import numpy

from cupy import util

from cupy.cuda import stream as stream_module

from cupy.core.core cimport ndarray

from cupy.core._carray cimport CArray
from cupy.core._carray cimport CIndexer
from cupy.core._carray cimport Indexer
from cupy.core.kernel_core cimport KernelGenerator
from cupy.core.kernel_core cimport ParameterInfo
from cupy.core.kernel_core cimport ParameterInfo_create_indexer
from cupy.core.kernel_core cimport ParameterInfo_parse
from cupy.core.kernel_core cimport RuntimeArgInfo
from cupy.core.kernel_core cimport RuntimeArgInfo_from_arg
from cupy.core.kernel_core cimport ParameterList

from cupy.core.kernel_core cimport preprocess_args as _preprocess_args
from cupy.core.kernel_core cimport reduce_dims as _reduce_dims
from cupy.core.kernel_core cimport get_dtype_name as _get_dtype_name
from cupy.core.kernel_core import parse_param_infos as _parse_param_infos
from cupy.core.kernel_core import decide_param_types as _decide_param_types
from cupy.core.kernel_core cimport do_broadcast as _do_broadcast
from cupy.core.kernel_core cimport get_out_args as _get_out_args
from cupy.core.kernel_core cimport guess_routine as _guess_routine


cdef class _BaseElementwiseKernelCallContext:
    cdef:
        readonly _BaseElementwiseKernel elementwise
        readonly object args
        readonly int size
        readonly object stream
        readonly str casting

    def __init__(self, _BaseElementwiseKernel elementwise, tuple args,
                 int size, stream, str casting):

        assert stream is None or isinstance(stream, stream_module.Stream), \
            type(stream)
        self.elementwise = elementwise
        self.args = args
        self.size = size
        self.stream = stream
        self.casting = casting

    cpdef call(self):

        size = self.size
        casting = self.casting
        elementwise = self.elementwise
        nin = elementwise.nin
        nout = elementwise.nout
        reduce_dims = elementwise.reduce_dims
        out_params = elementwise.out_params
        inout_params = elementwise.inout_params
        params = elementwise.params
        name = elementwise.name

        # Preprocess
        args = _preprocess_args(self.args)

        # Broadcast
        args, shape = _do_broadcast(args, inout_params, size)

        # Decide parameter dtypes.
        in_types, out_types = self.decide_param_types(args[:nin], args[nin:])

        in_args = [a if isinstance(a, ndarray) else t(a) for a, t in zip(args[:nin], in_types)]
        out_args = [a if isinstance(a, ndarray) else t(a) for a, t in zip(args[nin:], out_types)]

        # Allocate output args as needed.
        out_args = _get_out_args(
            out_args, out_types, shape, out_params, casting, size >= 0)
        if nout == 1:
            ret = out_args[0]
        else:
            ret = tuple(out_args)

        # If the shape is 0-sized, return immediately without any computation
        if 0 in shape:
            return ret

        inout_args = in_args + out_args

        # Reduce array dimensions
        if reduce_dims:
            inout_args, shape = _reduce_dims(inout_args, inout_params, shape)

        # Append indexer
        indexer = Indexer(shape)
        inout_args.append(indexer)

        inout_arg_infos = tuple([RuntimeArgInfo_from_arg(_)
                                 for _ in inout_args])

        key = inout_arg_infos
        kern = elementwise.kernel_cache.get(key)
        if kern is None:

            param_list = ParameterList(params, inout_arg_infos)

            # Retrieve source
            operation, preamble = self.get_code(param_list)

            # Launch kernel
            kern = _get_simple_elementwise_kernel(
                name, param_list, operation, preamble, None)

            elementwise.kernel_cache[key] = kern

        kern.linear_launch(indexer.size, inout_args, stream=self.stream)
        return ret

    cpdef decide_param_types(self, list in_args, list out_args):
        raise NotImplementedError()

    cpdef get_code(self, ParameterList param_list):
        raise NotImplementedError()


cdef class _ElementwiseKernelCallContext(_BaseElementwiseKernelCallContext):

    cdef:
         tuple _types

    def __init__(self, _BaseElementwiseKernel elementwise, tuple args, int size, stream):
        super(_ElementwiseKernelCallContext, self).__init__(
            elementwise, args, size, stream, None)

    cpdef decide_param_types(self, list in_args, list out_args):
        in_ndarray_types = tuple(
            [a.dtype.type if isinstance(a, ndarray) else None
             for a in in_args])
        out_ndarray_types = tuple([a.dtype.type for a in out_args])

        in_types, out_types, types = _decide_param_types(
            self.elementwise.in_params, self.elementwise.out_params,
            in_ndarray_types, out_ndarray_types)

        # Store types for succeeding get_code()
        self._types = types

        return in_types, out_types

    cpdef get_code(self, ParameterList param_list):
        types = self._types
        operation_ = self.elementwise.operation
        preamble_ = self.elementwise.preamble

        types_preamble = '\n'.join([
            'typedef %s %s;' % (_get_dtype_name(v), k) for k, v in types])
        preamble = types_preamble + '\n' + preamble_

        op = []
        for stmt in param_list.generate_ref_variable_decl_init_stmts():
            op.append(stmt)
        op.append(operation_)
        operation = '\n'.join(op)
        return operation, preamble


cdef class _BaseElementwiseKernel:

    def __init__(self, in_params, out_params, name, reduce_dims):
        if any([p.name == 'i' for p in in_params + out_params]):
            raise ValueError("Can not use 'i' as a parameter name")
        self.in_params = in_params
        self.out_params = out_params
        self.nin = len(in_params)
        self.nout = len(out_params)
        self.nargs = self.nin + self.nout
        self.inout_params = in_params + out_params
        self.params = self.inout_params + (ParameterInfo_parse('CIndexer _ind', False),)
        self.name = name
        self.reduce_dims = reduce_dims
        self.kernel_cache = {}

    cpdef call(self, args, kwargs):
        """Compiles and invokes the elementwise kernel.

        The compilation runs only if the kernel is not cached. Note that the
        kernels with different argument dtypes or dimensions are not
        compatible. It means that single ElementwiseKernel object may be
        compiled into multiple kernel binaries.

        Args:
            args: Arguments of the kernel.
            size (int): The size of index range. If specified, ``_ind.size()``
                in the kernel code will evaluate to this value. Otherwise,
                it's determined by the result of broadcast.

        Returns:
            Arrays are returned according to the ``out_params`` argument of the
            ``__init__`` method.

        """
        cdef _BaseElementwiseKernelCallContext call_ctx

        size = kwargs.pop('size', None)
        stream = kwargs.pop('stream', None)

        if size is None:
            size = -1

        call_ctx = self.create_call_context(args, size, stream, kwargs)

        return call_ctx.call()

    cpdef create_call_context(self, args, int size, stream, kwargs):
        raise NotImplementedError()


cdef class ElementwiseKernel(_BaseElementwiseKernel):

    """User-defined elementwise kernel.

    This class can be used to define an elementwise kernel with or without
    broadcasting.

    The kernel is compiled at an invocation of the
    :meth:`~ElementwiseKernel.__call__` method,
    which is cached for each device.
    The compiled binary is also cached into a file under the
    ``$HOME/.cupy/kernel_cache/`` directory with a hashed file name. The cached
    binary is reused by other processes.

    Args:
        in_params (str): Input argument list.
        out_params (str): Output argument list.
        operation (str): The body in the loop written in CUDA-C/C++.
        name (str): Name of the kernel function. It should be set for
            readability of the performance profiling.
        reduce_dims (bool): If ``False``, the shapes of array arguments are
            kept within the kernel invocation. The shapes are reduced
            (i.e., the arrays are reshaped without copy to the minimum
            dimension) by default. It may make the kernel fast by reducing the
            index calculations.
        options (list): Options passed to the ``nvcc`` command.
        preamble (str): Fragment of the CUDA-C/C++ code that is inserted at the
            top of the cu file.
        loop_prep (str): Fragment of the CUDA-C/C++ code that is inserted at
            the top of the kernel function definition and above the ``for``
            loop.
        after_loop (str): Fragment of the CUDA-C/C++ code that is inserted at
            the bottom of the kernel function definition.

    """

    def __init__(self, in_params, out_params, operation,
                 name='kernel', reduce_dims=True, preamble=''):
        in_params = _parse_param_infos(in_params, True)
        out_params = _parse_param_infos(out_params, False)
        super(ElementwiseKernel, self).__init__(in_params, out_params, name, reduce_dims)
        self.operation = operation
        self.preamble = preamble

    def __call__(self, *args, **kwargs):
        """Compiles and invokes the elementwise kernel.

        The compilation runs only if the kernel is not cached. Note that the
        kernels with different argument dtypes or dimensions are not
        compatible. It means that single ElementwiseKernel object may be
        compiled into multiple kernel binaries.

        Args:
            args: Arguments of the kernel.
            size (int): Range size of the indices. If specified, the variable
                ``n`` is set to this value. Otherwise, the result of
                broadcasting is used to determine the value of ``n``.

        Returns:
            Arrays are returned according to the ``out_params`` argument of the
            ``__init__`` method.

        """

        return self.call(args, kwargs)

    cpdef create_call_context(self, args, int size, stream, kwargs):
        if kwargs:
            raise TypeError('Wrong arguments %s' % kwargs)
        return _ElementwiseKernelCallContext(
            self, args, size, stream)


cdef class _UfuncKernelCallContext(_BaseElementwiseKernelCallContext):

    cdef:
        readonly object dtype
        readonly tuple ops
        readonly str _preamble

        str _routine
        tuple _in_types
        tuple _out_types


    def __init__(self, _BaseElementwiseKernel elementwise, tuple args,
                 int size, stream, tuple ops, dtype, str casting,
                 str preamble):

        super(_UfuncKernelCallContext, self).__init__(
            elementwise, args, size, stream, casting)

        self.dtype = dtype
        self.ops = ops
        self._preamble = preamble
        self._routine = None

    cpdef decide_param_types(self, list in_args, list out_args):
        in_types, out_types, routine = _guess_routine(
            self.elementwise.name,
            (<_UfuncKernel>self.elementwise)._routine_cache,
            self.ops, in_args, self.dtype)

        # Store variables for succeeding get_code()
        self._routine = routine
        self._in_types = in_types
        self._out_types = out_types

        return in_types, out_types

    cpdef get_code(self, ParameterList param_list):
        routine_ = self._routine
        preamble_ = self._preamble
        in_types = self._in_types
        out_types = self._out_types

        types = []
        op = []
        for i, x in enumerate(in_types):
            types.append('typedef %s in%d_type;' % (_get_dtype_name(x), i))
            if param_list.arg_infos[i].typ is ndarray:
                op.append(
                    'const in{0}_type in{0} = _raw_in{0}[_ind.get()];'.format(i))

        for i, x in enumerate(out_types):
            types.append('typedef %s out%d_type;' % (_get_dtype_name(x), i))
            op.append('{1} &out{0} = _raw_out{0}[_ind.get()];'.format(
                i, param_list.arg_infos[i + len(in_types)].get_dtype_name()))

        op.append(routine_)
        operation = '\n'.join(op)

        types.append(preamble_)
        preamble = '\n'.join(types)

        return operation, preamble


cdef class _UfuncKernel(_BaseElementwiseKernel):

    cdef:
        tuple _ops
        str _preamble
        dict _routine_cache

    def __init__(self, nin, nout, name, ops, preamble):
        self.nin = nin
        self.nout = nout
        self.nargs = nin + nout
        self._ops = ops
        self._preamble = preamble
        in_params = tuple(
            ParameterInfo_parse('T in%d' % i, True)
            for i in range(nin))
        out_params = tuple(
            ParameterInfo_parse('T out%d' % i, False)
            for i in range(nout))
        self.params = in_params + out_params + \
                      (ParameterInfo_parse('CIndexer _ind', False),)
        self._routine_cache = {}

        super(_UfuncKernel, self).__init__(in_params, out_params, name, False)


    cpdef create_call_context(self, args, int size, stream, kwargs):
        out = kwargs.pop('out', None)
        dtype = kwargs.pop('dtype', None)
        # Note default behavior of casting is 'same_kind' on numpy>=1.10
        casting = kwargs.pop('casting', 'same_kind')
        if dtype is not None:
            dtype = numpy.dtype(dtype).type
        if kwargs:
            raise TypeError('Wrong arguments %s' % kwargs)

        nargs = len(args)
        if nargs != self.nin and nargs != self.nargs:
            raise TypeError('Wrong number of arguments for %s' % self.name)

        if out is not None:
            if self.nout != 1:
                raise ValueError("Cannot use 'out' in %s" % self.name)
            if nargs != self.nin:
                raise ValueError("Cannot specify 'out' as both "
                                 "a positional and keyword argument")
            args += (out,)

        return _UfuncKernelCallContext(
            self, args, size, stream, self._ops, dtype, casting,
            self._preamble)


@util.memoize(for_each_device=True)
def _get_simple_elementwise_kernel(
        str name, ParameterList param_list, str operation, str preamble,
        frozenset kwargs):

    kwargs_ = dict(kwargs) if kwargs is not None else {}
    loop_prep = <str>kwargs_.pop('loop_prep', '')
    after_loop = <str>kwargs_.pop('after_loop', '')
    options = <tuple>kwargs_.pop('options', ())
    assert len(kwargs_) == 0

    gen = KernelGenerator(name)
    kern = gen.get_simple_elementwise_kernel(
        param_list, operation, preamble, loop_prep, after_loop, options)
    return kern


class ufunc(object):

    """Universal function.

    Arguments:
        name: Kernel name
        nin:
        nout:
        ops: A tuple which specifies the possible dtype combinations. Each
             element is a 3-element tuple, where first 2 elements are a tuples
             of input/output dtypes and the last element is the operation code.
        preamble:
        doc:

    Attributes:
        name (str): The name of the universal function.
        nin (int): Number of input arguments.
        nout (int): Number of output arguments.
        nargs (int): Number of all arguments.

    """
    def __init__(self, name, nin, nout, ops, preamble='', doc=''):
        self.name = name
        self.nin = nin
        self.nout = nout
        self.nargs = nin + nout
        self._ops = tuple(ops)
        assert len(preamble) == 0, preamble
        self.__doc__ = doc
        self.k = _UfuncKernel(nin, nout, name, tuple(ops), preamble)

    def __repr__(self):
        return "<ufunc '%s'>" % self.name

    @property
    def types(self):
        """A list of type signatures.

        Each type signature is represented by type character codes of inputs
        and outputs separated by '->'.

        """
        types = []
        for in_types, out_types, _ in self._ops:
            in_str = ''.join([<str>numpy.dtype(t).char for t in in_types])
            out_str = ''.join([<str>numpy.dtype(t).char for t in out_types])
            types.append('%s->%s' % (in_str, out_str))
        return types

    def __call__(self, *args, **kwargs):
        """Applies the universal function to arguments elementwise.

        Args:
            args: Input arguments. Each of them can be a :class:`cupy.ndarray`
                object or a scalar. The output arguments can be omitted or be
                specified by the ``out`` argument.
            out (cupy.ndarray): Output array. It outputs to new arrays
                default.
            dtype: Data type specifier.

        Returns:
            Output array or a tuple of output arrays.

        """

        return self.k.call(args, kwargs)


cpdef object create_ufunc(name, ops, routine=None, preamble='', doc=''):
    """ Creates ufunc instance.

    Arguments:
        name: Kernel name
        ops: A tuple which specifies the possible dtype combinations. Each
             element can be either a string which represents input-output
             dtype correspondence (e.g. ''bb->bb'), or a 2-element tuple
             in which the first element is a string described above, and the
             second element is the operation code for that dtype combination,
             which overrides `routine` argument.
        routine: Default operation code.
        preamble:
        doc:
    """
    _ops = []
    for t in ops:
        if isinstance(t, str):
            typ = t
            rt = routine
        elif isinstance(t, tuple):
            typ, rt = t
            assert isinstance(typ, str)
            assert isinstance(rt, str)
        else:
            assert False

        types = typ.split('->')
        if len(types) == 1:
            in_types = out_types = tuple(types)
        else:
            in_types, out_types = map(tuple, types)
        in_types = tuple([numpy.dtype(t).type for t in in_types])
        out_types = tuple([numpy.dtype(t).type for t in out_types])
        _ops.append((in_types, out_types, rt))

    return ufunc(name, len(_ops[0][0]), len(_ops[0][1]), _ops, preamble, doc)
