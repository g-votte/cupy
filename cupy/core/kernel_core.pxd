import numpy
import six

from cupy.cuda cimport function
from cupy.cuda.function cimport CPointer


cdef class ParameterInfo:
    cdef:
        readonly str name
        readonly object dtype
        readonly str ctype
        readonly bint raw
        readonly bint is_const

    cpdef get_var_name(self, RuntimeArgInfo arg_info)


cpdef ParameterInfo ParameterInfo_create_indexer(str name, bint raw)
cpdef ParameterInfo ParameterInfo_parse(str param, bint is_const)


cdef class RuntimeArgInfo:
    cdef:
        readonly object typ
        readonly object dtype
        readonly size_t ndim

    cpdef str get_base_type_expr(self)

    cpdef str get_dtype_name(self)


cpdef RuntimeArgInfo RuntimeArgInfo_from_arg(object arg)


cdef class ParameterList:
    cdef:
        readonly tuple params  # () of ParameterInfo
        readonly tuple arg_infos  # () of RuntimeArgInfo
        readonly tuple _var_names
        readonly tuple _base_types

    cdef tuple _ensure_var_names(self)

    cdef tuple _ensure_base_types(self)

    cdef list get_arrays(self)

    cdef str get_kernel_params_decl(self)

    cdef str get_entry_function_params_decl(self)

    cpdef str get_entry_function_param_list(self)

    cdef list generate_ref_variable_decl_init_stmts(self)


cdef class KernelGenerator(object):
    cdef:
        readonly str kernel_name
        object _s

    cpdef get_function(self, options)
    cpdef emit_elementwise_function(
            self, str class_name, ParameterList param_list, operation, preamble,
            loop_prep=*, after_loop=*)
    cpdef emit_reduction_function(
            self, str class_name, ParameterList param_list,
            int block_size, str reduce_type, object identity,
            str pre_map_expr, str reduce_expr, str post_map_expr,
            str type_preamble, str input_expr, str output_expr, str preamble)
    cpdef emit_kernel_entry_function(
            self, ParameterList param_list, str code)
    cdef emit_simple_elementwise_kernel_entry_function(
            self, ParameterList param_list, operation, preamble,
            loop_prep=*, after_loop=*)
    cpdef get_simple_elementwise_kernel(
            self, ParameterList param_list, operation, preamble,
            loop_prep=*, after_loop=*, tuple options=*)
    cdef emit_simple_reduction_kernel_entry_function(
            self, ParameterList param_list,
            int block_size, str reduce_type, object identity,
            str pre_map_expr, str reduce_expr, str post_map_expr,
            str type_preamble, str input_expr, str output_expr, str preamble)
    cpdef get_simple_reduction_kernel(
            self, ParameterList param_list,
            int block_size, str reduce_type, object identity,
            str pre_map_expr, str reduce_expr, str post_map_expr,
            str type_preamble, str input_expr, str output_expr, str preamble,
            tuple options=*)


cpdef function.Module compile_with_cache(
        str source, tuple options=*, arch=*, cachd_dir=*)


cpdef str get_dtype_name(dtype)


cpdef list preprocess_args(args)

cpdef tuple reduce_dims(list args, tuple params, tuple shape)

cpdef tuple do_broadcast(list args, tuple params, int size)

cdef list get_out_args(
        list out_args, tuple out_types, tuple out_shape, tuple out_params,
        str casting, bint use_size)

cdef tuple guess_routine(str name, dict cache, tuple ops, list in_args, dtype)
