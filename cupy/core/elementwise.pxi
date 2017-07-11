import string

import numpy
import six

from cupy import util

from cupy.cuda cimport device
from cupy.cuda cimport function
from cupy.cuda import stream as stream_module


cdef class KernelGenerator(object):
    cdef:
        readonly str kernel_name
        object _s

    def __init__(self, str kernel_name):
        self._s = six.StringIO()
        self.kernel_name = kernel_name

    cpdef get_function(self, options):
        code = self._s.getvalue()
        module = compile_with_cache(code, options)
        return module.get_function(self.kernel_name)

    cpdef emit_elementwise_function(
            self, str class_name, ParameterList param_list, operation, preamble,
            loop_prep='', after_loop=''):

        params_decl = param_list.get_entry_function_params_decl()
        self._s.write(string.Template('''

        class ${class_name} {
        private:
          ${preamble}
        public:
          __device__ void compute(${params_decl}) {
            ${loop_prep};
            CUPY_FOR(i, _ind.size()) {
              _ind.set(i);
              ${operation};
            }
            ${after_loop};
          }
        };
        ''').substitute(
            params_decl=params_decl,
            operation=operation,
            class_name=class_name,
            preamble=preamble,
            loop_prep=loop_prep,
            after_loop=after_loop))

    cpdef emit_reduction_function(
            self, str class_name, ParameterList param_list,
            int block_size, str reduce_type, object identity,
            str pre_map_expr, str reduce_expr, str post_map_expr,
            str type_preamble, str input_expr, str output_expr, str preamble):

        if identity is None:
            identity = ''
        else:
            identity = str(identity)

        params_decl = param_list.get_entry_function_params_decl()
        self._s.write(string.Template('''

        class ${class_name} {
        private:
          ${type_preamble}
          ${preamble}

          typedef ${reduce_type} _type_reduce;

          __device__ _type_reduce REDUCE(const _type_reduce& a,
                                         const _type_reduce& b) {
            return (${reduce_expr});
          }

          __device__ void _REDUCE(_type_reduce* _sdata,
                                  unsigned int tid,
                                  unsigned int offset) {
            if (tid < offset) {
              _type_reduce _a = _sdata[tid], _b = _sdata[(tid + offset)];
              _sdata[tid] = REDUCE(_a, _b);
            }
          }

        public:
          __device__ void compute(${params_decl}) {
            extern __shared__ _type_reduce _sdata_raw[];
            _type_reduce *_sdata = _sdata_raw;
            unsigned int _tid = threadIdx.x;

            int _J_offset = _tid / _block_stride;
            int _j_offset = _J_offset * _out_ind.size();
            int _J_stride = ${block_size};
            long long _j_stride = ${block_size}LL * _out_ind.size();

            for (int _i_base = blockIdx.x * _block_stride;
                 _i_base < _out_ind.size();
                 _i_base += gridDim.x * _block_stride) {
              _type_reduce _s = _type_reduce(${identity});
              int _i = _i_base + _tid % _block_stride;
              int _J = _J_offset;
              for (long long _j = _i + _j_offset; _j < _in_ind.size();
                   _j += _j_stride, _J += _J_stride) {
                _in_ind.set(_j);
                ${input_expr}
                _type_reduce _a = ${pre_map_expr};
                _s = REDUCE(_s, _a);
              }
              if (_block_stride < ${block_size}) {
                _sdata[_tid] = _s;
                __syncthreads();
                if (_block_stride <= 256) {
                  _REDUCE(_sdata, _tid, 256);
                  __syncthreads();
                  if (_block_stride <= 128) {
                    _REDUCE(_sdata, _tid, 128);
                    __syncthreads();
                    if (_block_stride <= 64) {
                      _REDUCE(_sdata, _tid, 64);
                      __syncthreads();
                      if (_block_stride <= 32) {
                        _REDUCE(_sdata, _tid, 32);
                        if (_block_stride <= 16) {
                          _REDUCE(_sdata, _tid, 16);
                          if (_block_stride <= 8) {
                            _REDUCE(_sdata, _tid, 8);
                            if (_block_stride <= 4) {
                              _REDUCE(_sdata, _tid, 4);
                              if (_block_stride <= 2) {
                                _REDUCE(_sdata, _tid, 2);
                                if (_block_stride <= 1) {
                                  _REDUCE(_sdata, _tid, 1);
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
                _s = _sdata[_tid];
                __syncthreads();
              }
              if (_J_offset == 0 && _i < _out_ind.size()) {
                _out_ind.set(_i);
                ${output_expr};
                {
                  _type_reduce a = _s;  // referred in 'post_map_expr'
                  ${post_map_expr};
                }
              }
            }
          }
        };''').substitute(
            class_name=class_name,
            params_decl=params_decl,
            block_size=block_size,
            reduce_type=reduce_type,
            identity=identity,
            reduce_expr=reduce_expr,
            pre_map_expr=pre_map_expr,
            post_map_expr=post_map_expr,
            type_preamble=type_preamble,
            input_expr=input_expr,
            output_expr=output_expr,
            preamble=preamble))

    cpdef emit_kernel_entry_function(
            self, ParameterList param_list, str code):

        kernel_params_decl = param_list.get_kernel_params_decl()
        self._s.write(string.Template('''
        // Kernel function
        extern "C" __global__ void ${kernel_name}(${kernel_params_decl}) {
          ${code}
        }
        ''').substitute(
            kernel_name=self.kernel_name,
            kernel_params_decl=kernel_params_decl,
            code=code))

    cdef emit_simple_elementwise_kernel_entry_function(
            self, ParameterList param_list, operation, preamble,
            loop_prep='', after_loop=''):

        # Emit elementwise function code
        class_name = self.kernel_name + '__impl'
        self.emit_elementwise_function(
            class_name, param_list, operation, preamble, loop_prep, after_loop)

        # Emit kernel entry function
        param_list_expr = param_list.get_entry_function_param_list()
        code = '{class_name}().compute({param_list_expr});'.format(
            class_name=class_name,
            param_list_expr=param_list_expr,
        )
        self.emit_kernel_entry_function(param_list, code)

    cpdef get_simple_elementwise_kernel(
            self, ParameterList param_list, operation, preamble,
            loop_prep='', after_loop='', tuple options=()):

        self.emit_simple_elementwise_kernel_entry_function(
            param_list, operation, preamble, loop_prep, after_loop)
        return self.get_function(options)

    cdef emit_simple_reduction_kernel_entry_function(
            self, ParameterList param_list,
            int block_size, str reduce_type, object identity,
            str pre_map_expr, str reduce_expr, str post_map_expr,
            str type_preamble, str input_expr, str output_expr, str preamble):

        # Emit reduction function code
        class_name = self.kernel_name + '__impl'
        self.emit_reduction_function(
            class_name, param_list, block_size, reduce_type, identity,
            pre_map_expr, reduce_expr, post_map_expr, type_preamble,
            input_expr, output_expr, preamble)

        # Emit kernel entry function
        param_list_expr = param_list.get_entry_function_param_list()
        code = '{class_name}().compute({param_list_expr});'.format(
            class_name=class_name,
            param_list_expr=param_list_expr,
        )
        self.emit_kernel_entry_function(param_list, code)

    cpdef get_simple_reduction_kernel(
            self, ParameterList param_list,
            int block_size, str reduce_type, object identity,
            str pre_map_expr, str reduce_expr, str post_map_expr,
            str type_preamble, str input_expr, str output_expr, str preamble,
            tuple options=()):

        self.emit_simple_reduction_kernel_entry_function(
            param_list, block_size, reduce_type, identity, pre_map_expr,
            reduce_expr, post_map_expr, type_preamble, input_expr, output_expr,
            preamble)
        return self.get_function(options)


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


cpdef str _get_dtype_name(dtype):
    if dtype is None:
        raise ValueError('dtype is None')
    name = _typenames.get(dtype)
    if name is None:
        name = _typenames[numpy.dtype(dtype).type]
    return name


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

    def __init__(self, str name, object dtype, str ctype, bint raw, bint is_const):
        self.name = name
        self.dtype = dtype
        self.ctype = ctype
        self.raw = raw
        self.is_const = is_const

    @staticmethod
    def indexer(str name, bint raw):
        return ParameterInfo(name, None, None, raw, False)

    @staticmethod
    def parse(str param, bint is_const):
        name = None
        dtype = None
        ctype = None
        raw = False

        s = tuple([i for i in param.split() if len(i) != 0])
        if len(s) < 2:
            raise Exception('Syntax error: %s' % param)

        for i in s[:-2]:
            if i == 'raw':
                raw = True
            else:
                raise Exception('Unknown keyword "%s"' % i)

        t, name = s[-2:]
        if t == 'CIndexer':
            return ParameterInfo.indexer(name, raw)
        else:
            if len(t) == 1:
                ctype = t
            else:
                dtype_ = numpy.dtype(t)
                dtype = dtype_.type
                if dtype_.name != t:
                    raise ValueError('Wrong type %s' % t)
                ctype = _get_dtype_name(dtype)

            return ParameterInfo(name, dtype, ctype, raw, is_const)

    def __repr__(self):
        return '<ParameterInfo name={!r} dtype={} ctype={} raw={} is_const={}>'.format(
            self.name, self.dtype, self.ctype, self.raw, self.is_const)

    cpdef get_var_name(self, RuntimeArgInfo arg_info):
        if not self.raw and arg_info.typ is ndarray:
            return '_raw_' + self.name
        else:
            return self.name

cdef class RuntimeArgInfo:
    cdef:
        readonly object typ
        readonly object dtype
        readonly size_t ndim

    def __init__(self, object typ, object dtype, size_t ndim):
        self.typ = typ
        self.dtype = dtype
        self.ndim = ndim

    @staticmethod
    def from_arg(arg):
        typ = type(arg)
        if typ is Indexer:
            dtype = None
            ndim = (<Indexer>arg).ndim
        else:
            dtype = arg.dtype.type
            ndim = arg.ndim
        return RuntimeArgInfo(typ, dtype, ndim)

    def __hash__(self):
        return hash(self.typ) ^ hash(self.dtype) ^ hash(self.ndim)

    def __richcmp__(RuntimeArgInfo x, RuntimeArgInfo y, int op):
        if op == 2:
            return x.typ == y.typ and x.dtype == y.dtype and x.ndim == y.ndim
        raise NotImplementedError()

    cpdef str get_base_type_expr(self):
        if self.typ is Indexer:
            t = 'CIndexer<%d>' % self.ndim
        else:
            dt = self.get_dtype_name()
            if self.typ is ndarray:
                t = 'CArray<%s, %d>' % (dt, self.ndim)
            else:
                t = dt
        return t

    cpdef str get_dtype_name(self):
        return _get_dtype_name(self.dtype)


cdef class ParameterList:
    cdef:
        readonly tuple params  # () of ParameterInfo
        readonly tuple arg_infos  # () of RuntimeArgInfo
        readonly tuple _var_names
        readonly tuple _base_types

    def __init__(self, tuple params, tuple arg_infos):
        assert len(params) == len(arg_infos), (len(params), len(arg_infos))
        assert all(isinstance(_, ParameterInfo) for _ in params)
        assert all(isinstance(_, RuntimeArgInfo) for _ in arg_infos)
        self.params = params
        self.arg_infos = arg_infos

        self._var_names = None
        self._base_types = None

    def __hash__(self):
        return hash(self.params) ^ hash(self.arg_infos)

    def __richcmp__(ParameterList x, ParameterList y, int op):
        if op == 2:
            return (x.params == y.params and
                    x.arg_infos == y.arg_infos)
        raise NotImplementedError()

    cdef tuple _ensure_var_names(self):
        cdef ParameterInfo p
        cdef RuntimeArgInfo a
        if self._var_names is not None:
            return
        ret = []
        for p, a in zip(self.params, self.arg_infos):
            ret.append(p.get_var_name(a))
        self._var_names = tuple(ret)

    cdef tuple _ensure_base_types(self):
        if self._base_types is not None:
            return
        ret = []
        for i in range(len(self.params)):
            arg_info = <RuntimeArgInfo>self.arg_infos[i]
            ret.append(arg_info.get_base_type_expr())
        self._base_types = tuple(ret)

    cdef list get_arrays(self):
        cdef ParameterInfo p
        cdef RuntimeArgInfo a

        return [p for p, a in zip(self.params, self.arg_infos)
                if not p.raw and a.typ is ndarray]

    cdef str get_kernel_params_decl(self):
        self._ensure_var_names()
        self._ensure_base_types()
        ret = []
        for i in range(len(self.params)):
            var_name = <str>(self._var_names[i])
            base_type = <str>(self._base_types[i])
            ret.append('%s %s' % (base_type, var_name))
        return ', '.join(ret)

    cdef str get_entry_function_params_decl(self):
        self._ensure_var_names()
        self._ensure_base_types()
        ret = []
        for i in range(len(self.params)):
            base_type = <str>(self._base_types[i])
            var_name = <str>(self._var_names[i])
            ret.append('%s %s' % (base_type, var_name))
        return ', '.join(ret)

    cpdef str get_entry_function_param_list(self):
        self._ensure_var_names()
        return ', '.join(self._var_names)

    cdef list generate_ref_variable_decl_init_stmts(self):
        cdef ParameterInfo p
        cdef RuntimeArgInfo a
        stmts = []
        for p, a in zip(self.params, self.arg_infos):
            if not p.raw and a.typ is ndarray:
                stmts.append(
                    '{t} &{n} = _raw_{n}[_ind.get()];'.format(
                        t=p.ctype, n=p.name))
        return stmts


@util.memoize()
def _parse_param_infos(str s, bint is_const):
    """Returns a tuple of ParameterInfo's specified by a string."""

    if len(s) == 0:
        return ()
    return tuple([ParameterInfo.parse(_, is_const) for _ in s.strip().split(',')])


@util.memoize()
def _decide_param_types(tuple in_params, tuple out_params, tuple in_arg_dtypes, tuple out_arg_dtypes):
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
    if params is None:
        values = args
    else:
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


cdef list _get_out_args(
        list out_args, tuple out_types, tuple out_shape, tuple out_params,
        str casting, bint use_size):
    """Allocates output arguments as needed."""
    cdef ParameterInfo p

    # There were no out args: allocate them.
    if len(out_args) == 0:
        # Check: if there is a raw parameter, size must be specified.
        if out_params is not None and not use_size:
            if any(p.raw for p in out_params):
                raise ValueError('Output array size is Undecided')
        return [ndarray(out_shape, t) for t in out_types]

    # There were out args: check dtype and shape consistency
    for a, p, t in zip(out_args, out_params, out_types):
        if not isinstance(a, ndarray):
            raise TypeError(
                'Output arguments type must be cupy.ndarray')
        if not p.raw and a.shape != out_shape:
            raise ValueError('Out shape is mismatched')

        if casting and not numpy.can_cast(t, a.dtype, casting=casting):
            msg = 'output (typecode \'{}\') could not be coerced to ' \
                  'provided output parameter (typecode \'{}\') according to ' \
                  'the casting rule "{}"'.format(
                      numpy.dtype(t).char,
                      a.dtype.char,
                      casting)
            raise TypeError(msg)

    return out_args


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
        args, shape = _broadcast(args, inout_params, size)

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

        inout_arg_infos = tuple([RuntimeArgInfo.from_arg(_)
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

    def __init__(self, in_params, out_params, name, reduce_dims):
        if any([p.name == 'i' for p in in_params + out_params]):
            raise ValueError("Can not use 'i' as a parameter name")
        self.in_params = in_params
        self.out_params = out_params
        self.nin = len(in_params)
        self.nout = len(out_params)
        self.nargs = self.nin + self.nout
        self.inout_params = in_params + out_params
        self.params = self.inout_params + (ParameterInfo.parse('CIndexer _ind', False),)
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

    cdef:
        readonly str operation
        readonly str preamble

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
            ParameterInfo.parse('T in%d' % i, True)
            for i in range(nin))
        out_params = tuple(
            ParameterInfo.parse('T out%d' % i, False)
            for i in range(nout))
        self.params = in_params + out_params + \
                      (ParameterInfo.parse('CIndexer _ind', False),)
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


cdef tuple _guess_routine_from_in_types(tuple ops, tuple in_types):
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


cdef tuple _guess_routine_from_dtype(tuple ops, object dtype):
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


cdef tuple _guess_routine(str name, dict cache, tuple ops, list in_args, dtype):
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


@util.memoize(for_each_device=True)
def _get_simple_elementwise_kernel(
        str name, ParameterList param_list, str operation, str preamble,
        frozenset kwargs):

    gen = KernelGenerator(name)
    kern = gen.get_simple_elementwise_kernel(
        param_list, operation, preamble, **dict(kwargs or {}))
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
        self._ops = ops
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
