# distutils: language = c++

import os
import six
import string

import numpy

from libcpp cimport vector

from cupy import cuda
from cupy.cuda cimport device
from cupy.core cimport core
from cupy.core.core import broadcast
from cupy.core.core cimport ndarray
from cupy.core._carray cimport Indexer
from cupy import util


cdef class ParameterInfo:

    def __init__(self, str name, object dtype, str ctype, bint raw, bint is_const):
        self.name = name
        self.dtype = dtype
        self.ctype = ctype
        self.raw = raw
        self.is_const = is_const

    def __repr__(self):
        return '<ParameterInfo name={!r} dtype={} ctype={} raw={} is_const={}>'.format(
            self.name, self.dtype, self.ctype, self.raw, self.is_const)

    cpdef get_var_name(self, RuntimeArgInfo arg_info):
        if not self.raw and arg_info.typ is core.ndarray:
            return '_raw_' + self.name
        else:
            return self.name

cpdef ParameterInfo ParameterInfo_create_indexer(str name, bint raw):
    return ParameterInfo(name, None, None, raw, False)


cpdef ParameterInfo ParameterInfo_parse(str param, bint is_const):
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
        return ParameterInfo_create_indexer(name, raw)
    else:
        if len(t) == 1:
            ctype = t
        else:
            dtype_ = numpy.dtype(t)
            dtype = dtype_.type
            if dtype_.name != t:
                raise ValueError('Wrong type %s' % t)
            ctype = get_dtype_name(dtype)

        return ParameterInfo(name, dtype, ctype, raw, is_const)


cdef class RuntimeArgInfo:

    def __init__(self, object typ, object dtype, size_t ndim):
        self.typ = typ
        self.dtype = dtype
        self.ndim = ndim

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
            if self.typ is core.ndarray:
                t = 'CArray<%s, %d>' % (dt, self.ndim)
            else:
                t = dt
        return t

    cpdef str get_dtype_name(self):
        return get_dtype_name(self.dtype)


cpdef RuntimeArgInfo RuntimeArgInfo_from_arg(object arg):
    typ = type(arg)
    if typ is Indexer:
        dtype = None
        ndim = (<Indexer>arg).ndim
    else:
        dtype = arg.dtype.type
        ndim = arg.ndim
    return RuntimeArgInfo(typ, dtype, ndim)


cdef class ParameterList:

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
                if not p.raw and a.typ is core.ndarray]

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
            if not p.raw and a.typ is core.ndarray:
                stmts.append(
                    '{t} &{n} = _raw_{n}[_ind.get()];'.format(
                        t=p.ctype, n=p.name))
        return stmts


cdef class KernelGenerator(object):

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


cdef str _header_source = None


cpdef str _get_header_source():
    global _header_source
    if _header_source is None:
        header_path = os.path.join(os.path.dirname(__file__), 'carray.cuh')
        with open(header_path) as header_file:
            _header_source = header_file.read()
    return _header_source


cpdef function.Module compile_with_cache(
        str source, tuple options=(), arch=None, cachd_dir=None):
    source = _get_header_source() + source
    return cuda.compile_with_cache(source, options, arch, cachd_dir)



cdef dict _python_type_to_numpy_type = {
    float: numpy.dtype(float).type,
    bool: numpy.dtype(bool).type}

for i in six.integer_types:
    _python_type_to_numpy_type[i] = numpy.int64


cdef str _all_type_chars = 'dfeqlihbQLIHB?'

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

cdef dict _typenames = {
    numpy.dtype(i).type: _typenames_base[numpy.dtype(i)]
    for i in _all_type_chars}


cdef tuple _python_scalar_type = six.integer_types + (float, bool)
cdef tuple _numpy_scalar_type = tuple([numpy.dtype(i).type
                                       for i in _all_type_chars])

cdef set _python_scalar_type_set = set(_python_scalar_type)
cdef set _numpy_scalar_type_set = set(_numpy_scalar_type)



cpdef str get_dtype_name(dtype):
    if dtype is None:
        raise ValueError('dtype is None')
    name = _typenames.get(dtype)
    if name is None:
        name = _typenames[numpy.dtype(dtype).type]
    return name


cpdef list preprocess_args(args):
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


cpdef tuple reduce_dims(list args, tuple params, tuple shape):
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


@util.memoize()
def parse_param_infos(str s, bint is_const):
    """Returns a tuple of ParameterInfo's specified by a string."""

    if len(s) == 0:
        return ()
    return tuple([ParameterInfo_parse(_, is_const) for _ in s.strip().split(',')])


@util.memoize()
def decide_param_types(tuple in_params, tuple out_params, tuple in_arg_dtypes, tuple out_arg_dtypes):
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


cpdef tuple do_broadcast(list args, tuple params, int size):
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


cdef list get_out_args(
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


cdef dict _kind_score = {
    'b': 0,
    'u': 1,
    'i': 1,
    'f': 2,
}


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


cdef tuple guess_routine(str name, dict cache, tuple ops, list in_args, dtype):
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
