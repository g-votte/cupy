import string

import numpy
import six

from cupy import util

from cupy.cuda cimport device
from cupy.cuda cimport function


cpdef _get_simple_elementwise_kernel(
        params, operation, name, preamble,
        loop_prep='', after_loop='', options=()):
    module_code = string.Template('''
    ${preamble}
    extern "C" __global__ void ${name}(${params}) {
      ${loop_prep};
      CUPY_FOR(i, _ind.size()) {
        _ind.set(i);
        ${operation};
      }
      ${after_loop};
    }
    ''').substitute(
        params=params,
        operation=operation,
        name=name,
        preamble=preamble,
        loop_prep=loop_prep,
        after_loop=after_loop)
    module = compile_with_cache(module_code, options)
    return module.get_function(name)


cdef dict _typenames_base = {
    numpy.dtype('float64'): 'double',
    numpy.dtype('float32'): 'float',
    numpy.dtype('float16'): 'float16',
    numpy.dtype('int64'): 'long long',
    numpy.dtype('int32'): 'int',
    numpy.dtype('int16'): 'short',
    numpy.dtype('int8'): 'signed char',
    numpy.dtype('uint64'): 'unsigned long long',
    numpy.dtype('uint32'): 'unsigned int',
    numpy.dtype('uint16'): 'unsigned short',
    numpy.dtype('uint8'): 'unsigned char',
    numpy.dtype('bool'): 'bool',
}

cdef str _all_type_chars = 'dfeqlihbQLIHB?'

cdef dict _typenames = {
    numpy.dtype(i).type: _typenames_base[numpy.dtype(i)]
    for i in _all_type_chars}

cdef tuple _python_scalar_type = six.integer_types + (float, bool)
cdef tuple _numpy_scalar_type = tuple([numpy.dtype(i).type
                                       for i in _all_type_chars])

cdef set _python_scalar_type_set = set(_python_scalar_type)
cdef set _numpy_scalar_type_set = set(_numpy_scalar_type)

cdef dict _kind_score = {
    'b': 0,
    'u': 1,
    'i': 1,
    'f': 2,
}


cdef dict _python_type_to_numpy_type = {
    float: numpy.dtype(float).type,
    bool: numpy.dtype(bool).type}
for i in six.integer_types:
    _python_type_to_numpy_type[i] = numpy.int64


cpdef str _get_typename(dtype):
    if dtype is None:
        raise ValueError('dtype is None')
    if dtype not in _typenames:
        dtype = numpy.dtype(dtype).type
    return _typenames[dtype]


cpdef list _preprocess_args(args):
    """Preprocesses arguments for kernel invocation

    - Checks device compatibility for ndarrays
    - Converts Python scalars into NumPy scalars
    """
    cdef list ret = []
    cdef int dev_id = device.get_device_id()
    cdef type typ

    for arg in args:
        typ = type(arg)
        if typ is ndarray:
            arr_dev = (<ndarray?>arg).data.device
            if arr_dev is not None and arr_dev.id != dev_id:
                raise ValueError(
                    'Array device must be same as the current '
                    'device: array device = %d while current = %d'
                    % (arr_dev.id, dev_id))
        elif typ in _python_scalar_type_set:
            arg = _python_type_to_numpy_type[typ](arg)
        elif typ in _numpy_scalar_type_set:
            pass
        else:
            raise TypeError('Unsupported type %s' % typ)
        ret.append(arg)
    return ret


cpdef tuple _get_args_info(list args):
    ret = []
    for a in args:
        t = type(a)
        if t is Indexer:
            dtype = None
        else:
            dtype = a.dtype.type
        ret.append((t, dtype, a.ndim))
    return tuple(ret)


cpdef tuple _reduce_dims(list args, tuple params, tuple shape):
    """Reduce the dimensions of arrays into the minimum without copy."""
    cdef Py_ssize_t i, j, n, ndim, cnt, axis, s
    cdef vector.vector[Py_ssize_t] vecshape, newshape, newstrides
    cdef vector.vector[bint] is_array_flags
    cdef vector.vector[vector.vector[Py_ssize_t]] args_strides
    cdef ParameterInfo p
    cdef ndarray arr, view
    cdef bint is_array

    ndim = len(shape)
    if ndim <= 1:
        return args, shape

    n = len(args)
    for p, a in zip(params, args):
        is_array = not p.raw and isinstance(a, ndarray)
        is_array_flags.push_back(is_array)
        if is_array:
            arr = a
            args_strides.push_back(arr._strides)

    vecshape = shape
    axis = -1
    cnt = 0
    for i in range(1, ndim):
        if vecshape[i - 1] == 1:
            continue
        for j in range(<Py_ssize_t>args_strides.size()):
            if args_strides[j][i] * vecshape[i] != args_strides[j][i - 1]:
                cnt += 1
                axis = i - 1
                break
        else:
            vecshape[i] *= vecshape[i - 1]
            vecshape[i - 1] = 1
    if vecshape[ndim - 1] != 1:
        cnt += 1
        axis = ndim - 1

    if cnt == ndim:
        return args, shape
    if cnt == 1:
        newshape.assign(<Py_ssize_t>1, <Py_ssize_t>vecshape[axis])
        ret = []
        for is_array, a in zip(is_array_flags, args):
            if is_array:
                arr = (<ndarray>a).view()
                newstrides.assign(
                    <Py_ssize_t>1, <Py_ssize_t>arr._strides[axis])
                arr._set_shape_and_strides(newshape, newstrides, False)
                a = arr
            ret.append(a)
        return ret, tuple(newshape)

    for i in range(ndim):
        if vecshape[i] != 1:
            newshape.push_back(vecshape[i])

    ret = []
    for i, a in enumerate(args):
        if is_array_flags[i]:
            arr = a
            arr = arr.view()
            newstrides.clear()
            for i in range(ndim):
                if vecshape[i] != 1:
                    newstrides.push_back(arr._strides[i])
            arr._set_shape_and_strides(newshape, newstrides, False)
            a = arr
        ret.append(a)
    return ret, tuple(newshape)


cdef class ParameterInfo:
    cdef:
        readonly str name
        readonly object dtype
        readonly str ctype
        readonly bint raw
        readonly bint is_const

    def __init__(self, str param, bint is_const):
        self.name = None
        self.dtype = None
        self.ctype = None
        self.raw = False
        self.is_const = is_const
        s = tuple([i for i in param.split() if len(i) != 0])
        if len(s) < 2:
            raise Exception('Syntax error: %s' % param)

        t, self.name = s[-2:]
        if t == 'CIndexer':
            pass
        elif len(t) == 1:
            self.ctype = t
        else:
            dtype = numpy.dtype(t)
            self.dtype = dtype.type
            if dtype.name != t:
                raise ValueError('Wrong type %s' % t)
            self.ctype = _get_typename(self.dtype)

        for i in s[:-2]:
            if i == 'raw':
                self.raw = True
            else:
                raise Exception('Unknown keyword "%s"' % i)


cdef class ParameterList:
    cdef:
        readonly tuple params
        readonly tuple infos
        readonly tuple _var_names
        readonly tuple _base_types

    def __init__(self, tuple params, list args):
        assert len(params) == len(args)
        self.params = params
        self.infos = self._get_infos(args)

        self._var_names = None
        self._base_types = None

    def __hash__(self):
        return hash(self.params) ^ hash(self.infos)

    def __richcmp__(ParameterList x, ParameterList y, int op):
        if op == 2:
            return (x.params == y.params and
                    x.infos == y.infos)
        raise NotImplementedError()

    cdef tuple _get_infos(self, list args):
        ret = []
        for a in args:
            t = type(a)
            if t is Indexer:
                dtype = None
            else:
                dtype = a.dtype.type
            ret.append((t, dtype, a.ndim))
        return tuple(ret)

    cdef tuple _ensure_var_names(self):
        cdef ParameterInfo p
        cdef tuple a
        if self._var_names is not None:
            return
        ret = []
        for p, a in zip(self.params, self.infos):
            is_array = a[0] is ndarray
            if is_array and not p.raw:
                ret.append('_raw_' + p.name)
            else:
                ret.append(p.name)
        self._var_names = tuple(ret)

    cdef tuple _ensure_base_types(self):
        if self._base_types is not None:
            return
        ret = []
        for i in range(len(self.params)):
            p = <ParameterInfo>(self.params[i])
            type, dtype, ndim = <tuple>(self.infos[i])
            is_array = type is ndarray
            if type is Indexer:
                t = 'CIndexer<%d>' % ndim
            else:
                t = _get_typename(dtype)
                if is_array:
                    t = 'CArray<%s, %d>' % (t, ndim)
            ret.append(t)
        self._base_types = tuple(ret)

    cdef list get_arrays(self):
        cdef ParameterInfo p
        cdef tuple a

        return [p for p, a in zip(self.params, self.infos)
                if not p.raw and a[0] is ndarray]

    cdef str get_kernel_params_decl(self):
        self._ensure_var_names()
        self._ensure_base_types()
        ret = []
        for i in range(len(self.params)):
            var_name = <str>(self._var_names[i])
            base_type = <str>(self._base_types[i])
            ret.append('%s %s' % (base_type, var_name))
        return ', '.join(ret)

    cdef str get_reduction_function_params_decl(self):
        self._ensure_var_names()
        self._ensure_base_types()
        ret = []
        for i in range(len(self.params)):
            base_type = <str>(self._base_types[i])
            var_name = <str>(self._var_names[i])
            ret.append('%s %s' % (base_type, var_name))
        return ', '.join(ret)

    cdef str get_reduction_function_param_list(self):
        self._ensure_var_names()
        return ', '.join(self._var_names)

    cdef list generate_ref_variable_decl_init_stmts(self):
        cdef ParameterInfo p
        cdef tuple a
        stmts = []
        for p, a in zip(self.params, self.infos):
            if not p.raw and a[0] is ndarray:
                if p.is_const:
                    fmt = '{t} &{n} = _raw_{n}[_ind.get()];'
                else:
                    fmt = '{t} &{n} = _raw_{n}[_ind.get()];'
                stmts.append(fmt.format(t=p.ctype, n=p.name))
        return stmts


@util.memoize()
def _get_param_info(s, is_const):
    if len(s) == 0:
        return ()
    return tuple([ParameterInfo(i, is_const) for i in s.strip().split(',')])


@util.memoize()
def _decide_param_types(in_params, out_params, in_arg_dtypes, out_arg_dtypes):
    """Determines the dtypes of input/output arguments in the generated kernel.

    Args:
        in_params: ParameterInfo's of input arguments.
        out_params: ParameterInfo's of output arguments..
        in_arg_dtypes: Dtypes of input arguments.
        out_arg_dtypes: Dtypes of output arguments.

    Returns:
        A 3-element tuple where the first 2 elements are the tuples of
        input/output dtypes in the generated kernel corresponding to each
        input/output argument, and the last element is a tuple containing the
        unique pairs of (dtype, ctype) in undefined order.
    """
    type_dict = {}
    if out_arg_dtypes:
        assert len(out_params) == len(out_arg_dtypes)
        for p, a in zip(out_params, out_arg_dtypes):
            if a is None:
                raise TypeError('Output arguments must be cupy.ndarray')
            if p.dtype is not None:
                if numpy.dtype(a) != numpy.dtype(p.dtype):
                    raise TypeError(
                        'Type is mismatched. %s %s %s' % (p.name, a, p.dtype))
            elif p.ctype in type_dict:
                t = type_dict[p.ctype]
                if numpy.dtype(t) != numpy.dtype(a):
                    raise TypeError(
                        'Type is mismatched. %s %s %s %s' % (
                            p.name, a, t, p.ctype))
            else:
                type_dict[p.ctype] = a

    assert len(in_params) == len(in_arg_dtypes)
    unknown_ctype = []
    for p, a in zip(in_params, in_arg_dtypes):
        if a is None:
            if p.dtype is None:
                unknown_ctype.append(p.ctype)
        else:
            if p.dtype is not None:
                if numpy.dtype(a) != numpy.dtype(p.dtype):
                    raise TypeError(
                        'Type is mismatched. %s %s %s' % (p.name, a, p.dtype))
            elif p.ctype in type_dict:
                t = type_dict[p.ctype]
                if numpy.dtype(t) != numpy.dtype(a):
                    raise TypeError(
                        'Type is mismatched. %s %s %s %s' % (
                            p.name, a, t, p.ctype))
            else:
                type_dict[p.ctype] = a

    in_types = tuple([type_dict[p.ctype] if p.dtype is None else p.dtype
                      for p in in_params])
    out_types = tuple([type_dict[p.ctype] if p.dtype is None else p.dtype
                       for p in out_params])
    return in_types, out_types, tuple(type_dict.items())


cdef tuple _broadcast(list args, tuple params, int size):
    cpdef Py_ssize_t i
    cpdef ParameterInfo p
    cpdef bint has_non_none
    cpdef bint use_size = size >= 0
    values = []
    has_non_none = False
    for i in range(len(args)):
        p = params[i]
        a = args[i]
        if not p.raw and isinstance(a, ndarray):
            has_non_none = True
            values.append(a)
        else:
            values.append(None)

    if use_size:
        if has_non_none:
            raise ValueError("Specified 'size' can be used only "
                             "if all of the ndarray are 'raw'.")
    else:
        if not has_non_none:
            raise ValueError('Loop size is Undecided')

    brod = broadcast(*values)
    values = list(brod.values)
    for i in range(len(args)):
        if values[i] is None:
            values[i] = args[i]

    if use_size:
        shape = size,
    else:
        shape = brod.shape
    return values, shape


cdef list _get_out_args(list out_args, tuple out_types, tuple out_shape,
                        str casting):
    if not out_args:
        return [ndarray(out_shape, t) for t in out_types]

    for i, a in enumerate(out_args):
        if not isinstance(a, ndarray):
            raise TypeError(
                'Output arguments type must be cupy.ndarray')
        if a.shape != out_shape:
            raise ValueError('Out shape is mismatched')
        out_type = out_types[i]
        if not numpy.can_cast(out_type, a.dtype, casting=casting):
            msg = 'output (typecode \'{}\') could not be coerced to ' \
                  'provided output parameter (typecode \'{}\') according to ' \
                  'the casting rule "{}"'.format(
                      numpy.dtype(out_type).char,
                      a.dtype.char,
                      casting)
            raise TypeError(msg)
    return out_args


cdef list _get_out_args_with_params(
        list out_args, tuple out_types, tuple out_shape, tuple out_params,
        bint use_size):
    """Allocates output arguments as needed."""
    cdef ParameterInfo p

    # There were no out args: allocate them.
    if len(out_args) == 0:
        # Check: if there is a raw parameter, size must be specified.
        for p in out_params:
            if p.raw and not use_size:
                raise ValueError('Output array size is Undecided')
        return [ndarray(out_shape, t) for t in out_types]

    # There were out args: check dtype and shape consistency
    for a, p in zip(out_args, out_params):
        if not isinstance(a, ndarray):
            raise TypeError(
                'Output arguments type must be cupy.ndarray')
        if not p.raw and a.shape != out_shape:
            raise ValueError('Out shape is mismatched')

    return out_args


@util.memoize(for_each_device=True)
def _get_elementwise_kernel(ParameterList param_list, types, operation, name,
                            preamble, kwargs):
    kernel_params = param_list.get_kernel_params_decl()
    types_preamble = '\n'.join(
        'typedef %s %s;' % (_get_typename(v), k) for k, v in types)
    preamble = types_preamble + '\n' + preamble

    op = []
    for stmt in param_list.generate_ref_variable_decl_init_stmts():
        op.append(stmt)
    op.append(operation)
    operation = '\n'.join(op)
    return _get_simple_elementwise_kernel(
        kernel_params, operation, name,
        preamble, **dict(kwargs))


cdef class ElementwiseKernel:

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

    cdef:
        readonly tuple in_params
        readonly tuple out_params
        readonly Py_ssize_t nin
        readonly Py_ssize_t nout
        readonly Py_ssize_t nargs
        readonly tuple params
        readonly str operation
        readonly str name
        readonly bint reduce_dims
        readonly str preamble
        readonly object kwargs

    def __init__(self, in_params, out_params, operation,
                 name='kernel', reduce_dims=True, preamble='', **kwargs):
        self.in_params = _get_param_info(in_params, True)
        self.out_params = _get_param_info(out_params, False)
        self.nin = len(self.in_params)
        self.nout = len(self.out_params)
        self.nargs = self.nin + self.nout
        param_rest = _get_param_info('CIndexer _ind', False)
        self.params = self.in_params + self.out_params + param_rest
        self.operation = operation
        self.name = name
        self.reduce_dims = reduce_dims
        self.preamble = preamble
        self.kwargs = frozenset(kwargs.items())
        names = [p.name for p in self.in_params + self.out_params]
        if 'i' in names:
            raise ValueError("Can not use 'i' as a parameter name")

    def __call__(self, *args, **kwargs):
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

        cdef function.Function kern

        size = kwargs.pop('size', None)
        stream = kwargs.pop('stream', None)
        if kwargs:
            raise TypeError('Wrong arguments %s' % kwargs)

        n_args = len(args)
        if n_args != self.nin and n_args != self.nargs:
            raise TypeError('Wrong number of arguments for %s' % self.name)

        if size is None:
            size = -1

        # Preprocess
        args = _preprocess_args(args)

        # Broadcast
        args, shape = _broadcast(args, self.params, size)
        in_args = args[:self.nin]
        out_args = args[self.nin:]

        # Decide parameter dtypes.
        in_ndarray_types = tuple(
            [a.dtype.type if isinstance(a, ndarray) else None
             for a in in_args])
        out_ndarray_types = tuple([a.dtype.type for a in out_args])

        in_types, out_types, types = _decide_param_types(
            self.in_params, self.out_params,
            in_ndarray_types, out_ndarray_types)

        # Allocate output args as needed.
        out_args = _get_out_args_with_params(
            out_args, out_types, shape, self.out_params, size is not None)
        if self.nout == 1:
            ret = out_args[0]
        else:
            ret = tuple(out_args)

        # If the shape is 0-sized, return immediately without any computation
        if 0 in shape:
            return ret

        inout_args = in_args + out_args

        # Reduce array dimensions
        if self.reduce_dims:
            inout_args, shape = _reduce_dims(inout_args, self.params, shape)

        # Append indexer
        indexer = Indexer(shape)
        inout_args.append(indexer)

        # Compile and launch the kernel
        param_list = ParameterList(self.params, inout_args)
        kern = _get_elementwise_kernel(
            param_list, types, self.operation,
            self.name, self.preamble, self.kwargs)
        kern.linear_launch(indexer.size, inout_args, shared_mem=0,
                           block_max_size=128, stream=stream)
        return ret


@util.memoize(for_each_device=True)
def _get_ufunc_kernel(
        in_types, out_types, routine, ParameterList param_list, name,
        preamble):
    kernel_params = param_list.get_kernel_params_decl()

    types = []
    op = []
    for i, x in enumerate(in_types):
        types.append('typedef %s in%d_type;' % (_get_typename(x), i))
        if param_list.infos[i][0] is ndarray:
            op.append(
                'const in{0}_type in{0} = _raw_in{0}[_ind.get()];'.format(i))

    for i, x in enumerate(out_types):
        types.append('typedef %s out%d_type;' % (_get_typename(x), i))
        op.append('{1} &out{0} = _raw_out{0}[_ind.get()];'.format(
            i, _get_typename(param_list.infos[i + len(in_types)][1])))

    op.append(routine)
    operation = '\n'.join(op)

    types.append(preamble)
    preamble = '\n'.join(types)

    return _get_simple_elementwise_kernel(
        kernel_params, operation, name, preamble)


cdef tuple _guess_routine_from_in_types(list ops, tuple in_types):
    cdef Py_ssize_t i, n
    cdef tuple op, op_types
    n = len(in_types)
    can_cast = numpy.can_cast
    for op in ops:
        op_types = op[0]
        for i in range(n):
            if not can_cast(in_types[i], op_types[i]):
                break
        else:
            return op
    return None


cdef tuple _guess_routine_from_dtype(list ops, object dtype):
    cdef tuple op, op_types
    for op in ops:
        op_types = op[1]
        for t in op_types:
            if t != dtype:
                break
        else:
            return op
    return None


cdef bint _check_should_use_min_scalar(list in_args) except *:
    cdef int kind, max_array_kind, max_scalar_kind
    cdef bint all_scalars
    all_scalars = True
    max_array_kind = -1
    max_scalar_kind = -1
    for i in in_args:
        kind = _kind_score[i.dtype.kind]
        if isinstance(i, ndarray):
            all_scalars = False
            max_array_kind = max(max_array_kind, kind)
        else:
            max_scalar_kind = max(max_scalar_kind, kind)
    return (max_scalar_kind != -1 and
            not all_scalars and
            max_array_kind >= max_scalar_kind)


cdef tuple _guess_routine(str name, dict cache, list ops, list in_args, dtype):
    """Find the best-matching operation from given dtype or input arguments.

    Args:
        name: ufunc name. Just used in error message.
        cache: Cache
        ops: List of candidate oprations.
        in_args: Input arguments.
        dtype: dtype

    Returns:
        One of the elements in op argument; a 3-element tuple where the first 2
        elements are input/output dtypes and the last element is the operation
        code.
    """
    if dtype is None:
        # dtype is not given. Guess operation from input arguments.
        use_raw_value = _check_should_use_min_scalar(in_args)
        if use_raw_value:
            in_types = tuple(in_args)
            op = None
        else:
            in_types = tuple([i.dtype.type for i in in_args])
            op = cache.get(in_types)

        if op is None:
            # Not found in cache
            op = _guess_routine_from_in_types(ops, in_types)
            if not use_raw_value:
                cache[in_types] = op
    else:
        # dtype is given. Guess operation from dtype.
        op = cache.get(dtype)
        if op is None:
            # Not found in cache
            op = _guess_routine_from_dtype(ops, dtype)
            cache[dtype] = op

    if op is not None:
        return op
    raise TypeError('Wrong type of arguments for %s' % name)


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
        self._ops = ops
        self._preamble = preamble
        self.__doc__ = doc
        _in_params = tuple(
            ParameterInfo('T in%d' % i, True)
            for i in range(nin))
        _out_params = tuple(
            ParameterInfo('T out%d' % i, False)
            for i in range(nout))
        self._params = _in_params + _out_params + (
            ParameterInfo('CIndexer _ind', False),)
        self._routine_cache = {}

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

        cdef function.Function kern

        out = kwargs.pop('out', None)
        dtype = kwargs.pop('dtype', None)
        # Note default behavior of casting is 'same_kind' on numpy>=1.10
        casting = kwargs.pop('casting', 'same_kind')
        if dtype is not None:
            dtype = numpy.dtype(dtype).type
        if kwargs:
            raise TypeError('Wrong arguments %s' % kwargs)

        n_args = len(args)
        if n_args != self.nin and n_args != self.nargs:
            raise TypeError('Wrong number of arguments for %s' % self.name)

        args = _preprocess_args(args)
        if out is None:
            in_args = args[:self.nin]
            out_args = args[self.nin:]
        else:
            if self.nout != 1:
                raise ValueError("Cannot use 'out' in %s" % self.name)
            if n_args != self.nin:
                raise ValueError("Cannot specify 'out' as both "
                                 "a positional and keyword argument")

            in_args = list(args)
            out_args = _preprocess_args((out,))
            args += out_args

        broad = broadcast(*args)
        shape = broad.shape

        in_types, out_types, routine = _guess_routine(
            self.name, self._routine_cache, self._ops, in_args, dtype)

        out_args = _get_out_args(out_args, out_types, shape, casting)
        if self.nout == 1:
            ret = out_args[0]
        else:
            ret = tuple(out_args)

        if 0 in shape:
            return ret

        inout_args = []
        for i, t in enumerate(in_types):
            x = broad.values[i]
            inout_args.append(x if isinstance(x, ndarray) else t(x))
        inout_args.extend(out_args)
        inout_args, shape = _reduce_dims(inout_args, self._params, shape)
        indexer = Indexer(shape)
        inout_args.append(indexer)
        param_list = ParameterList(self._params, inout_args)

        kern = _get_ufunc_kernel(
            in_types, out_types, routine, param_list,
            self.name, self._preamble)

        kern.linear_launch(indexer.size, inout_args)
        return ret


cpdef create_ufunc(name, ops, routine=None, preamble='', doc=''):
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
        if not isinstance(t, tuple):
            typ = t
            rt = routine
        else:
            typ, rt = t

        types = typ.split('->')
        if len(types) == 1:
            in_types = out_types = tuple(types)
        else:
            in_types, out_types = map(tuple, types)
        in_types = tuple([numpy.dtype(t).type for t in in_types])
        out_types = tuple([numpy.dtype(t).type for t in out_types])
        _ops.append((in_types, out_types, rt))

    return ufunc(name, len(_ops[0][0]), len(_ops[0][1]), _ops, preamble, doc)
