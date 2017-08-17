import string

import numpy
import six

from cupy import util

from cupy.cuda cimport device
from cupy.cuda cimport function
from cupy.cuda import stream as stream_module


def _str_format(str s, **kwargs):
    return string.Template(s).substitute(**kwargs)


cdef class _IndentedConstructContext(object):
    cdef:
        readonly KernelCodeEmitter emitter
        readonly int indent
        readonly str starting_code
        readonly str ending_code

    def __init__(self, KernelCodeEmitter emitter, str starting_code, str ending_code, int indent):
        self.emitter = emitter
        self.indent = indent
        self.starting_code = starting_code
        self.ending_code = ending_code

    def __enter__(self):
        self.emitter.emit_line(self.starting_code)
        self.emitter.indent += self.indent
        return self

    def __exit__(self, type, value, traceback):
        self.emitter.indent -= self.indent
        self.emitter.emit_line(self.ending_code)


cdef class KernelCodeEmitter(object):
    cdef:
        readonly str kernel_name
        int indent
        object _s

    def __init__(self, str kernel_name):
        self._s = six.StringIO()
        self.kernel_name = kernel_name

    cpdef write(self, str code):
        self._s.write(code)

    cpdef emit_line(self, str line, int indent=0):
        s = self._s
        s.write(' ' * (self.indent + indent))
        s.write(line)
        s.write('\n')

    cpdef emit_lines(self, str lines, int indent=0):
        s = self._s
        ind = ' ' * (self.indent + indent)
        for line in lines.split():
            s.write(ind)
            s.write(line)
            s.write('\n')

    cdef _IndentedConstructContext indented_construct(
            self, str starting_code, str ending_code, indent=2):

        return _IndentedConstructContext(self, starting_code, ending_code, indent)

    cpdef get_function(self, options):
        code = self._s.getvalue()
        module = compile_with_cache(code, options)
        return module.get_function(self.kernel_name)

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
        self.write(string.Template('''

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

    cpdef _IndentedConstructContext construct_kernel_entry_function(
            self, ParameterList param_list):

        kernel_params_decl = param_list.get_kernel_params_decl()
        self.emit_line('// Kernel function')
        return self.indented_construct(
            _str_format(
                'extern "C" __global__ void ${kernel_name}(${kernel_params_decl}) {',
                kernel_name=self.kernel_name,
                kernel_params_decl=kernel_params_decl),
            '}')

    cpdef emit_kernel_entry_function(
            self, ParameterList param_list, str code):

        with self.construct_kernel_entry_function(param_list):
            self.emit_line(code)

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
    numpy.dtype('complex128'): 'complex<double>',
    numpy.dtype('complex64'): 'complex<float>',
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

cdef str _all_type_chars = 'dfDFeqlihbQLIHB?'
# for c in 'dDfFeqlihbQLIHB?':
#    print('#', c, '...', np.dtype(c).name)
# d ... float64
# D ... complex128
# f ... float32
# F ... complex64
# e ... float16
# q ... int64
# l ... int64
# i ... int32
# h ... int16
# b ... int8
# Q ... uint64
# L ... uint64
# I ... uint32
# H ... uint16
# B ... uint8
# ? ... bool

cdef dict _typenames = {
    numpy.dtype(i).type: _typenames_base[numpy.dtype(i)]
    for i in _all_type_chars}

cdef tuple _python_scalar_type = six.integer_types + (float, bool, complex)
cdef tuple _numpy_scalar_type = tuple([numpy.dtype(i).type
                                       for i in _all_type_chars])

cdef set _python_scalar_type_set = set(_python_scalar_type)
cdef set _numpy_scalar_type_set = set(_numpy_scalar_type)

cdef dict _kind_score = {
    'b': 0,
    'u': 1,
    'i': 1,
    'f': 2,
    'c': 3,
}


cdef dict _python_type_to_numpy_type = {
    float: numpy.dtype(float).type,
    complex: numpy.dtype(complex).type,
    bool: numpy.dtype(bool).type}
for i in six.integer_types:
    _python_type_to_numpy_type[i] = numpy.int64


cdef str _get_dtype_name(dtype):
    if dtype is None:
        raise ValueError('dtype is None')
    name = _typenames.get(dtype)
    if name is None:
        name = _typenames[numpy.dtype(dtype).type]
    return name


cdef list _preprocess_args(args):
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
            arr_dev = (<ndarray>arg).data.device
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
        for i, a in enumerate(args):
            if is_array_flags[i]:
                arr = a
                arr = arr.view()
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

        s = [i for i in param.split() if len(i) != 0]
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

    cdef get_var_name(self, ArgInfo arg_info):
        if not self.raw and arg_info.typ is ndarray:
            return '_raw_' + self.name
        else:
            return self.name


cdef class ArgInfo:
    cdef:
        readonly object arg
        readonly object typ
        readonly object dtype
        readonly tuple shape
        readonly Py_ssize_t ndim
        readonly bint is_ndarray
        readonly tuple strides

    def __init__(self, object arg, type typ, dtype, tuple shape, int ndim, tuple strides):
        self.arg = arg
        self.typ = typ
        self.dtype = dtype
        self.shape = shape
        self.ndim = ndim
        self.strides = strides
        self.is_ndarray = typ is ndarray

    def __repr__(self):
        return '<ArgInfo typ={} shape={} dtype={} ndim={}>'.format(
            self.typ.__name__,
            self.shape,
            'None' if self.dtype is None else self.dtype.name,
            self.ndim)

    def __hash__(self):
        return (hash(self.typ) ^ hash(self.dtype) ^ hash(self.shape)
                ^ hash(self.arg) ^ hash(self.strides))

    def __richcmp__(ArgInfo x, ArgInfo y, int op):
        if op == 2:
            if x is y:
                return True
            return (
                x.arg == y.arg and
                x.typ is y.typ and
                x.dtype == y.dtype and
                x.shape == y.shape and
                x.strides == y.strides)
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

    cdef str get_dtype_name(self):
        return _get_dtype_name(self.dtype)


cpdef ArgInfo ArgInfo_from_arg(arg, bint hold_strides=False):
    typ = type(arg)
    strides = None

    # Holds scalar value, which is used in _guess_routine()
    arg_ = None

    if arg is None:
        dtype = None
        shape = None
        ndim = -1
    elif typ is ndarray:
        dtype = (<ndarray>arg).dtype
        # (<ndarray>arg).shape incurs a symbolic lookup
        shape = tuple((<ndarray>arg)._shape)
        ndim = len(shape)
        if hold_strides:
            strides = tuple((<ndarray>arg)._strides)
    elif typ is Indexer:
        dtype = None
        shape = (<Indexer>arg).shape
        ndim = len(shape)
    elif typ is slice:
        dtype = None
        shape = None
        ndim = -1
    elif typ in _python_scalar_type_set:
        arg_ = arg
        dtype = None
        shape = ()
        ndim = 0
    elif typ in _numpy_scalar_type_set:
        arg_ = arg
        dtype = arg.dtype
        shape = arg.shape
        ndim = len(shape)
    else:
        dtype = arg.dtype
        shape = arg.shape
        ndim = len(shape)
        if hold_strides:
            strides = arg.strides

    return ArgInfo(arg_, typ, dtype, shape, ndim, strides)


cpdef list ArgInfo_from_args(args, bint hold_strides=False):
    return [ArgInfo_from_arg(arg, hold_strides) for arg in args]


cdef class ParameterList:
    cdef:
        readonly tuple params  # () of ParameterInfo
        readonly tuple arg_infos  # () of ArgInfo
        readonly tuple _var_names
        readonly tuple _base_types
        readonly int nin
        readonly int nout

    def __init__(self, tuple params, tuple arg_infos, int nin, int nout):
        assert len(params) == len(arg_infos), (len(params), len(arg_infos))
        assert nin >= 0, nin
        assert nout >= 0, nout
        assert nin + nout <= len(params), (nin, nout, len(params))
        self.params = params
        self.arg_infos = arg_infos
        self.nin = nin
        self.nout = nout

        self._var_names = None
        self._base_types = None

    def __hash__(self):
        return hash(self.params) ^ hash(self.arg_infos)

    def __richcmp__(ParameterList x, ParameterList y, int op):
        if op == 2:
            return (x.params == y.params and
                    x.arg_infos == y.arg_infos)
        raise NotImplementedError()

    cpdef tuple get_in_pair(self, in_index):
        assert in_index < self.nin
        return self.params[in_index], self.arg_infos[in_index]

    cpdef tuple get_out_pair(self, out_index):
        assert out_index < self.nout
        return self.params[self.nin + out_index], self.arg_infos[self.nin + out_index]

    cdef tuple var_names(self):
        self._ensure_var_names()
        return self._var_names

    cdef _ensure_var_names(self):
        cdef ParameterInfo p
        cdef ArgInfo a
        if self._var_names is not None:
            return
        ret = []
        for p, a in zip(self.params, self.arg_infos):
            ret.append(p.get_var_name(a))
        self._var_names = tuple(ret)

    cdef tuple base_types(self):
        self._ensure_base_types()
        return self._base_types

    cdef _ensure_base_types(self):
        if self._base_types is not None:
            return
        ret = []
        for i in range(len(self.params)):
            arg_info = <ArgInfo>self.arg_infos[i]
            ret.append(arg_info.get_base_type_expr())
        self._base_types = tuple(ret)

    cdef list get_arrays(self):
        cdef ParameterInfo p
        cdef ArgInfo a

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
        cdef ArgInfo a
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


cdef list _preprocess_broadcast(list arg_infos, tuple params):
    cdef Py_ssize_t i
    cdef ParameterInfo p
    cdef ArgInfo a
    arg_infos_ = []
    for p, a in zip(params, arg_infos):
        if not p.raw and a.is_ndarray:
            arg_infos_.append(a)
        else:
            arg_infos_.append(None)

    return arg_infos_

cdef _broadcast_impl _broadcast(list arg_infos, tuple params, int size):
    cdef Py_ssize_t i
    cdef ParameterInfo p
    cdef bint has_non_none
    cdef bint use_size = size >= 0
    cdef ArgInfo a
    if params is not None:
        arg_infos = _preprocess_broadcast(arg_infos, params)
        has_non_none = any([_ is not None for _ in arg_infos])
        if use_size:
            if has_non_none:
                raise ValueError("Specified 'size' can be used only "
                                 "if all of the ndarray are 'raw'.")
        else:
            if not has_non_none:
                raise ValueError('Loop size is Undecided')

    brod_impl = _broadcast_impl(arg_infos)
    return brod_impl


cdef list _get_out_args(
        list out_args, tuple out_types, tuple out_shape, tuple out_params,
        str casting, bint use_size):
    """Allocates output arguments as needed."""
    cdef bint raw

    # There were no out args: allocate them.
    if len(out_args) == 0:
        # Check: if there is a raw parameter, size must be specified.
        if out_params is not None and not use_size:
            if any(p.raw for p in out_params):
                raise ValueError('Output array size is Undecided')
        return [ndarray(out_shape, t) for t in out_types]

    # There were out args: check dtype and shape consistency
    for i, (a, t) in enumerate(zip(out_args, out_types)):
        if not isinstance(a, ndarray):
            raise TypeError(
                'Output arguments type must be cupy.ndarray')
        if out_params is None:
            raw = False
        else:
            raw = (<ParameterInfo>out_params[i]).raw
        if not raw and a.shape != out_shape:
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


cdef class _BaseKernelCallContext(object):
    cdef:
        readonly _BaseKernel kernel

    def __init__(self, _BaseKernel kernel):
        self.kernel = kernel

    cdef decide_param_types(self, list in_arg_infos, list out_arg_infos):
        raise NotImplementedError()

    cpdef broadcast_and_cast(self, arg_infos, size):
        kernel = self.kernel
        nin = kernel.nin
        inout_params = kernel.inout_params

        key = ('broadcast', kernel, size, tuple(arg_infos))
        tup = kernel.kernel_cache.get(key)
        if tup is None:

            # Broadcast
            brod_impl = _broadcast(arg_infos, inout_params, size)
            shape = (size,) if size >= 0 else brod_impl.shape()

            arg_infos_ = brod_impl.apply_infos(arg_infos)

            kernel.kernel_cache[key] = (brod_impl, shape, arg_infos_)
        else:
            brod_impl, shape, arg_infos_ = tup

        # Decide parameter dtypes.
        in_types, out_types = self.decide_param_types(
            arg_infos_[:nin],
            arg_infos_[nin:])

        return brod_impl, shape, arg_infos_, in_types, out_types


cdef class _BaseElementwiseKernelCallContext(_BaseKernelCallContext):
    cdef:
        readonly arg_infos
        readonly int size
        readonly str casting
        readonly int nin
        readonly int nout
        readonly object stream

    def __init__(self, _BaseElementwiseKernel kernel, arg_infos,
                 int size, str casting, stream):

        assert stream is None or isinstance(stream, stream_module.Stream), \
            type(stream)

        super(_BaseElementwiseKernelCallContext, self).__init__(kernel)

        self.arg_infos = arg_infos
        self.size = size
        self.casting = casting
        self.nin = kernel.nin
        self.nout = kernel.nout
        self.stream = stream

    cpdef call(self, args):
        size = self.size
        casting = self.casting
        stream = self.stream
        kernel = self.kernel
        nin = kernel.nin
        nout = kernel.nout
        reduce_dims = kernel.reduce_dims
        out_params = kernel.out_params
        inout_params = kernel.inout_params
        params = kernel.params
        name = kernel.name

        # Preprocess
        args = _preprocess_args(args)

        # Broadcast
        arg_infos = ArgInfo_from_args(args, True)
        tup = self.broadcast_and_cast(arg_infos, size)
        brod_impl, shape, arg_infos_, in_types, out_types = tup

        args = brod_impl.apply(args)

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

        inout_arg_infos = tuple(ArgInfo_from_args(inout_args))

        key = ('kernel', device.get_device_id(), inout_arg_infos) + \
              self.get_code_args()
        kern = kernel.kernel_cache.get(key)
        if kern is None:

            param_list = ParameterList(params, inout_arg_infos, nin, nout)
            params_decl = param_list.get_entry_function_params_decl()

            #
            emitter = KernelCodeEmitter(name)
            op_class_name = name + '__'
            loop_class_name = name + '__loop_'

            # Emit elementwise op class
            self.emit_op_class(emitter, op_class_name, param_list)

            # Emit elementwise loop computation class
            self.emit_loop_class(
                emitter, loop_class_name, op_class_name, param_list,
                '', '', '')

            # Emit elementwise computation class
            self.emit_kernel_entry_function(emitter, loop_class_name, param_list)

            kern = emitter.get_function(())
            kernel.kernel_cache[key] = kern

        kern.linear_launch(indexer.size, inout_args, stream=stream)
        return ret

    cpdef emit_call_stmt(self, emitter, param_list):
        pass

    cdef tuple get_code_args(self):
        return ()

    cdef get_code(self, ParameterList param_list, tuple args):
        raise NotImplementedError()

    cpdef get_op_param_list(self, ParameterList param_list):
        lst = ('i',) + \
              param_list._var_names
        return ', '.join(lst)

    cpdef get_op_params_decl(self, ParameterList param_list):
        lst = ['const ptrdiff_t& i'] + \
              ['{}& {}'.format(base_type, var_name)
               for base_type, var_name
               in zip(param_list.base_types(), param_list.var_names())]
        return ', '.join(lst)

    cpdef emit_op_class(self, KernelCodeEmitter emitter, class_name, ParameterList param_list):
        # Retrieve source
        operation, preamble = self.get_code(param_list, self.get_code_args())

        op_params_decl = self.get_op_params_decl(param_list)

        emitter.write(string.Template('''
class ${class_name} {
private:
  ${preamble}
public:
  __device__ void op(${op_params_decl}) {
      ${operation};
  }
};
''').substitute(
            op_params_decl=op_params_decl,
            operation=operation,
            class_name=class_name,
            preamble=preamble))

    cdef emit_kernel_entry_function(
            self, KernelCodeEmitter emitter,
            str loop_class_name,
            ParameterList param_list):

        # Emit kernel entry function
        with emitter.construct_kernel_entry_function(param_list):
            param_list_expr = param_list.get_entry_function_param_list()
            emitter.emit_line(
                _str_format(
                    '${loop_class_name}().compute(${param_list_expr});',
                    loop_class_name=loop_class_name,
                    param_list_expr=param_list_expr))

    cpdef emit_op_call_stmt(
            self, KernelCodeEmitter emitter, str class_name,
            ParameterList param_list):

        op_param_list = self.get_op_param_list(param_list)
        emitter.emit_line(
            '{}().op({});'.format(class_name, op_param_list))

    cpdef emit_loop_class(
            self, KernelCodeEmitter emitter,
            str class_name, str op_class_name, ParameterList param_list,
            str preamble,
            str loop_prep='', str after_loop=''):

        starting_code = _str_format(
            '''
class ${class_name} {
private:
  ${preamble}
public:
  __device__ void compute(${params_decl}) {
    ${loop_prep};
    CUPY_FOR(i, _ind.size()) {
      _ind.set(i);
''',
            params_decl=param_list.get_kernel_params_decl(),
            class_name=class_name,
            preamble=preamble,
            loop_prep=loop_prep)

        ending_code = _str_format(
            '''
    }
    ${after_loop};
  }
};
''',
            after_loop=after_loop)

        params_decl = param_list.get_entry_function_params_decl()
        with emitter.indented_construct(starting_code, ending_code, 6):
            self.emit_op_call_stmt(emitter, op_class_name, param_list)


cdef class _ElementwiseKernelCallContext(_BaseElementwiseKernelCallContext):

    cdef:
         tuple _types

    def __init__(
            self, _BaseElementwiseKernel elementwise, arg_infos,
            int size, stream):

        super(_ElementwiseKernelCallContext, self).__init__(
            elementwise, arg_infos, size, None, stream)

    cdef decide_param_types(self, list in_arg_infos, list out_arg_infos):
        cdef ArgInfo a
        kernel = <_ElementwiseKernel>self.kernel

        key = (
            'param_types',
            self.kernel, tuple(in_arg_infos), tuple(out_arg_infos))
        tup = kernel.kernel_cache.get(key)
        if tup is None:
            in_ndarray_types = tuple(
                [a.dtype.type if a.is_ndarray else None for a in in_arg_infos])
            out_ndarray_types = tuple([a.dtype.type for a in out_arg_infos])

            tup = _decide_param_types(
                kernel.in_params, kernel.out_params,
                in_ndarray_types, out_ndarray_types)

            kernel.kernel_cache[key] = tup

        in_types, out_types, types = tup

        # Store types for succeeding get_code()
        self._types = types

        return in_types, out_types

    cdef tuple get_code_args(self):
        return (self._types,)

    cdef get_code(self, ParameterList param_list, tuple args):
        cdef tuple types
        types, = args
        operation_ = (<_ElementwiseKernel>self.kernel).operation
        preamble_ = (<_ElementwiseKernel>self.kernel).preamble

        types_preamble = '\n'.join([
            'typedef %s %s;' % (_get_dtype_name(v), k) for k, v in types])
        preamble = types_preamble + '\n' + preamble_

        op = []
        for stmt in param_list.generate_ref_variable_decl_init_stmts():
            op.append(stmt)
        op.append(operation_)
        operation = '\n'.join(op)
        return operation, preamble


cdef class _BaseKernel(object):

    def __init__(self, in_params, out_params, name):
        self.in_params = in_params
        self.out_params = out_params
        self.nin = len(in_params)
        self.nout = len(out_params)
        self.nargs = self.nin + self.nout
        self.inout_params = in_params + out_params
        self.name = name
        self.kernel_cache = {}


cdef class _BaseElementwiseKernel(_BaseKernel):
    cdef:
        readonly tuple params
        readonly bint reduce_dims

    def __init__(self, in_params, out_params, name, reduce_dims):
        cdef ParameterInfo p
        if any([p.name == 'i' for p in in_params]):
            raise ValueError("Can not use 'i' as a parameter name")
        if any([p.name == 'i' for p in out_params]):
            raise ValueError("Can not use 'i' as a parameter name")
        super(_BaseElementwiseKernel, self).__init__(in_params, out_params, name)
        self.inout_params = in_params + out_params
        self.params = self.inout_params + (ParameterInfo.parse('CIndexer _ind', False),)
        self.reduce_dims = reduce_dims

    cpdef _BaseElementwiseKernelCallContext create_call_context(
            self, arg_infos, kwargs):

        raise NotImplementedError()

cdef class _ElementwiseKernel(_BaseElementwiseKernel):

    cdef:
        readonly str operation
        readonly str preamble

    def __init__(
            self, str in_params, str out_params, str operation,
            str name='kernel', bint reduce_dims=True, str preamble=''):
        in_params_ = _parse_param_infos(in_params, True)
        out_params_ = _parse_param_infos(out_params, False)
        super(_ElementwiseKernel, self).__init__(
            in_params_, out_params_, name, reduce_dims)
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
            size (int): The size of index range. If specified, ``_ind.size()``
                in the kernel code will evaluate to this value. Otherwise,
                it's determined by the result of broadcast.

        Returns:
            Arrays are returned according to the ``out_params`` argument of the
            ``__init__`` method.

        """

        arg_infos = ArgInfo_from_args(args)
        call_ctx = self.create_call_context(arg_infos, kwargs)
        return call_ctx.call(args)

    cpdef _BaseElementwiseKernelCallContext create_call_context(
            self, arg_infos, kwargs):

        size = kwargs.pop('size', None)
        stream = kwargs.pop('stream', None)

        nargs = len(arg_infos)
        if nargs != self.nin and nargs != self.nargs:
            raise TypeError('Wrong number of arguments for %s' % self.name)

        if size is None:
            size = -1

        return _ElementwiseKernelCallContext(
            self, arg_infos, size, stream)


cdef class ElementwiseKernel(object):

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

    _cache = {}

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

        readonly _ElementwiseKernel _kernel

    def __init__(
            self, str in_params, str out_params, str operation,
            str name='kernel', bint reduce_dims=True, str preamble=''):

        cdef _ElementwiseKernel kernel

        key = (in_params, out_params, operation, name, reduce_dims, preamble)
        kernel = self._cache.get(key)
        if kernel is None:
            kernel = _ElementwiseKernel(
                in_params, out_params, operation, name, reduce_dims, preamble)
            self._cache[key] = kernel

        self.in_params = kernel.in_params
        self.out_params = kernel.out_params
        self.nin = kernel.nin
        self.nout = kernel.nout
        self.nargs = kernel.nargs
        self.params = kernel.params
        self.operation = kernel.operation
        self.name = kernel.name
        self.reduce_dims = kernel.reduce_dims
        self.preamble = kernel.preamble
        self._kernel = kernel


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

        return self._kernel(*args, **kwargs)


cdef class _UfuncKernelCallContext(_BaseElementwiseKernelCallContext):

    cdef:
        readonly object dtype

        str _routine
        tuple _in_types
        tuple _out_types

    def __init__(
            self, _BaseElementwiseKernel elementwise, arg_infos,
            dtype, str casting):

        super(_UfuncKernelCallContext, self).__init__(
            elementwise, arg_infos, -1, casting, None)

        self.dtype = dtype
        self._routine = None

    cdef decide_param_types(self, list in_arg_infos, list out_arg_infos):
        kernel = <_UfuncKernel>self.kernel

        key = (
            'param_types',
            tuple(in_arg_infos), tuple(out_arg_infos), self.dtype)
        tup = kernel.kernel_cache.get(key)
        if tup is None:
            tup = _guess_routine(
                kernel.name,
                kernel._routine_cache,
                kernel._ops,
                in_arg_infos, self.dtype)
            kernel.kernel_cache[key] = tup

        in_types, out_types, routine = tup

        # Store variables for succeeding get_code()
        self._routine = routine
        self._in_types = in_types
        self._out_types = out_types

        return in_types, out_types

    cdef tuple get_code_args(self):
        return (self._routine, self._in_types, self._out_types)

    cdef get_code(self, ParameterList param_list, tuple args):
        cdef ArgInfo a
        preamble_ = (<_UfuncKernel>self.kernel)._preamble
        routine_, in_types, out_types = args

        types = []
        op = []
        for i, x in enumerate(in_types):
            types.append('typedef %s in%d_type;' % (_get_dtype_name(x), i))
            a = param_list.arg_infos[i]
            if a.is_ndarray:
                op.append(
                    'const in{0}_type in{0}(_raw_in{0}[_ind.get()]);'.format(i))

        for i, x in enumerate(out_types):
            types.append('typedef %s out%d_type;' % (_get_dtype_name(x), i))
            a = param_list.arg_infos[i + len(in_types)]
            op.append('{1} &out{0} = _raw_out{0}[_ind.get()];'.format(
                i, a.get_dtype_name()))

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

        super(_UfuncKernel, self).__init__(in_params, out_params, name, True)

    def __call__(self, *args, **kwargs):
        out = kwargs.pop('out', None)
        if out is not None:
            if self.nout != 1:
                raise ValueError("Cannot use 'out' in %s" % self.name)
            if len(args) != self.nin:
                raise ValueError("Cannot specify 'out' as both "
                                 "a positional and keyword argument")
            args += (out,)

        arg_infos = ArgInfo_from_args(args)
        call_ctx = self.create_call_context(arg_infos, kwargs)
        return call_ctx.call(args)

    cpdef _BaseElementwiseKernelCallContext create_call_context(
            self, arg_infos, kwargs):

        dtype = kwargs.pop('dtype', None)
        # Note default behavior of casting is 'same_kind' on numpy>=1.10
        casting = kwargs.pop('casting', 'same_kind')
        if dtype is not None and not isinstance(dtype, numpy.dtype):
            dtype = numpy.dtype(dtype)
        if kwargs:
            raise TypeError('Wrong arguments %s' % kwargs)

        return _UfuncKernelCallContext(
            self, arg_infos, dtype, casting)


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
            if t != dtype.type:
                break
        else:
            return op
    return None


cdef bint _check_should_use_min_scalar(list in_arg_infos) except *:
    cdef int kind, max_array_kind, max_scalar_kind
    cdef bint all_scalars
    cdef ArgInfo a
    all_scalars = True
    max_array_kind = -1
    max_scalar_kind = -1
    for a in in_arg_infos:
        kind = _kind_score[a.dtype.kind]
        if a.is_ndarray:
            all_scalars = False
            max_array_kind = max(max_array_kind, kind)
        else:
            max_scalar_kind = max(max_scalar_kind, kind)
    return (max_scalar_kind != -1 and
            not all_scalars and
            max_array_kind >= max_scalar_kind)


cdef tuple _guess_routine(
        str name, dict cache, tuple ops, list in_arg_infos, dtype):
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
    cdef ArgInfo a

    if dtype is None:
        # dtype is not given. Guess operation from input arguments.
        use_raw_value = _check_should_use_min_scalar(in_arg_infos)
        if use_raw_value:
            in_types = tuple([a.dtype.type if a.is_ndarray else a.arg for a in in_arg_infos])
            op = None
        else:
            in_types = tuple([a.dtype.type for a in in_arg_infos])
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
    if dtype is None:
        dtype = tuple([i.dtype.type for i in in_args])
    raise TypeError('Wrong type (%s) of arguments for %s' %
                    (dtype, name))


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

        return self.k(*args, **kwargs)


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
