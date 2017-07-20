import functools
import six
from six.moves import builtins
import string
import threading
import warnings

import numpy

from cupy.core import core
from cupy.core import kernel_core
from cupy.core import _carray
from cupy import creation
from cupy import logic
from cupy import math
from cupy import sorting
from cupy import statistics
from cupy import util


_thread_local = threading.local()


class _FusionNode(object):

    def __init__(self, var, creator_op):
        assert isinstance(var, _FusionVar)
        assert creator_op is None or isinstance(creator_op, (FusionOp, ReductionOp))
        self.var = var
        self.creator_op = creator_op

    @property
    def dtype(self):
        return self.var.dtype

    def __repr__(self):
        return "<_FusionNode {:x}, var={!r} creator_op={!r}]>".format(
            id(self), self.var, self.creator_op)

class FusionOp(object):

    def __init__(self, name, operation, param_names,
                 nin, nout, in_nodes, out_nodes, types):
        self.name = name
        self.operation = operation
        self.param_names = param_names
        self.nin = nin
        self.nout = nout
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.types = types
        assert len(in_nodes) == nin  # TODO: remove nin & nout parameter?
        assert len(out_nodes) == nout

    def __repr__(self):
        return "<FusionOp, name={}, types=[{}]>".format(
            self.name, ', '.join(_.name for _ in self.types))

    def build_kernel_name(self):
        return self.name + '_' + '_'.join([
            'IN_' + '_'.join(build_kernel_name(_) for _ in self.in_nodes),
            'OUT_' + '_'.join(build_kernel_name(_) for _ in self.out_nodes),
        ])


class ReductionOp(object):

    def __init__(self, name, routine, param_names, in_node, out_node, types, identity):
        assert isinstance(routine, tuple), type(routine)
        assert len(routine) >= 3, len(routine)
        self.name = name
        self.routine = routine
        self.param_names = param_names
        self.in_node = in_node
        self.out_node = out_node
        self.in_nodes = [in_node]
        self.out_nodes = [out_node]
        self.types = types
        self.identity = identity

    def build_kernel_name(self):
        return self.name


class _FusionVar(object):

    def __init__(self, dtype, const=None):
        assert isinstance(dtype, numpy.dtype), type(dtype)
        self.dtype = dtype
        self.const = const
        self.ctype = _dtype_to_ctype[dtype]

    def __repr__(self):
        return "<_FusionVar {:x}, dtype={}, const={}>".format(
            id(self), self.dtype, self.const)

    def build_kernel_name(self):
        return self.dtype.name


class _FusionRef(object):

    def __init__(self, node):
        assert isinstance(node, _FusionNode)
        self.node = node

    @property
    def dtype(self):
        return self.node.var.dtype

    def __repr__(self):
        return "<_FusionRef %x, dtype=%s>" % (id(self), self.dtype)

    def build_kernel_name(self):
        return build_kernel_name(self._var)

    def __neg__(self):
        return negative(self)

    def __add__(self, other):
        return add(self, other)

    def __iadd__(self, other):
        return add(self, other, self)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return subtract(self, other)

    def __isub__(self, other):
        return subtract(self, other, self)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other):
        return multiply(self, other)

    def __imul__(self, other):
        return multiply(self, other, self)

    def __rmul__(self, other):
        return multiply(other, self)

    def __div__(self, other):
        return divide(self, other)

    def __idiv__(self, other):
        return divide(self, other, self)

    def __rdiv__(self, other):
        return divide(other, self)

    def __truediv__(self, other):
        return true_divide(self, other)

    def __itruediv__(self, other):
        return true_divide(self, other, self)

    def __rtruediv__(self, other):
        return true_divide(other, self)

    def __floordiv__(self, other):
        return floor_divide(self, other)

    def __ifloordiv__(self, other):
        return floor_divide(self, other, self)

    def __rfloordiv__(self, other):
        return floor_divide(other, self)

    def __mod__(self, other):
        return remainder(self, other)

    def __imod__(self, other):
        return remainder(self, other, self)

    def __rmod__(self, other):
        return remainder(other, self)

    def __pow__(x, y):
        return power(x, y)

    def __ipow__(self, other):
        return power(self, other, self)

    def __lshift__(self, other):
        return left_shift(self, other)

    def __ilshift__(self, other):
        return left_shift(self, other, self)

    def __rlshift__(self, other):
        return left_shift(other, self)

    def __rshift__(self, other):
        return right_shift(self, other)

    def __irshift__(self, other):
        return right_shift(self, other, self)

    def __rrshift__(self, other):
        return right_shift(other, self)

    def __and__(self, other):
        return bitwise_and(self, other)

    def __iand__(self, other):
        return bitwise_and(self, other, self)

    def __rand__(self, other):
        return bitwise_and(other, self)

    def __or__(self, other):
        return bitwise_or(self, other)

    def __ior__(self, other):
        return bitwise_or(self, other, self)

    def __ror__(self, other):
        return bitwise_or(other, self)

    def __xor__(self, other):
        return bitwise_xor(self, other)

    def __ixor__(self, other):
        return bitwise_xor(self, other, self)

    def __rxor__(self, other):
        return bitwise_xor(other, self)

    def __invert__(self):
        return invert(self)

    def __lt__(self, other):
        return less(self, other)

    def __le__(self, other):
        return less_equal(self, other)

    def __eq__(self, other):
        return equal(self, other)

    def __ne__(self, other):
        return not_equal(self, other)

    def __gt__(self, other):
        return greater(self, other)

    def __ge__(self, other):
        return greater_equal(self, other)

    def __nonzero__(self):
        raise Exception("Can't cast to bool")

    def __bool__(self):
        raise Exception("Can't cast to bool")

    def copy(self, *args, **kwargs):
        return copy(self, *args, **kwargs)

    def sum(self, *args, **kwargs):
        return sum(self, *args, **kwargs)


class _FusionCodeGenerator:
    def __init__(self, kernel_name, in_refs, out_refs):
        in_nodes = [ref.node for ref in in_refs]
        out_nodes = [ref.node for ref in out_refs]

        self._gen = kernel_core.KernelGenerator(kernel_name)
        self.fusers = []

        self._out_vars = [_.var for _ in out_nodes]
        self._in_vars = [_.var for _ in in_nodes]

        #-------------------
        # Prepare arguments
        #-------------------

        # Allocate output args
        out_vars = []
        for i, var in enumerate(self._out_vars):
            if var not in self._in_vars:
                out_vars.append(var)


        func_var_list = list(self._in_vars) + out_vars

        # ParameterInfo
        param_infos = []
        for i, var in enumerate(func_var_list):
            #var_name = self._var_name_map[id(var)]
            var_name = 'vvv{}'.format(i)
            param_info = kernel_core.ParameterInfo(var_name, var.dtype, var.ctype, False, False)
            param_infos.append(param_info)

        assert len(func_var_list) == len(param_infos), \
            (len(func_var_list), len(param_infos))

        # Append indexer
        param_infos.append(kernel_core.ParameterInfo_create_indexer('_ind', False))

        self.param_infos = param_infos

        self.subgraphs = self.decompose_into_subgraphs(in_nodes, out_nodes)

        for i, subgraph in enumerate(self.subgraphs):
            is_elmwise, in_nodes_, out_nodes_ = subgraph
            print(is_elmwise)
            for n in in_nodes_:
                print("   ", n)
            print("-")
            for n in out_nodes_:
                print("   ", n)

            if is_elmwise:
                name = 'fusion_elmementwise_{}'.format(i)
                fuser = _FusionElementwiseCodeGenerator(name, in_nodes_, out_nodes_)
            else:
                name = 'fusion_reduction_{}'.format(i)
                fuser = _FusionReductionCodeGenerator(name, in_nodes_, out_nodes_)
            self.fusers.append(fuser)

    def decompose_into_subgraphs(self, in_nodes, out_nodes):
        n_colors = 0
        node_color_map = {}

        out_nodes_list = []  # List of out nodes for each color
        in_nodes_list = []  # List of in nodes for each color
        is_elmwise_list = []  # List of <bool>is_elmwise

        def find_upstream_color(node, is_elmwise):
            col = node_color_map.get(id(node))
            if col is not None:
                return col

            op = node.creator_op
            if op is not None:
                if isinstance(op, FusionOp) == is_elmwise:
                    for n in op.in_nodes:
                        col = find_upstream_color(n, is_elmwise)
                        if col is not None:
                            return col

            return None

        def fill_upstream_color(node, color, is_elmwise, boundary_nodes):
            op = node.creator_op
            if op is not None:
                if isinstance(op, FusionOp) == is_elmwise:
                    node_color_map[id(node)] = color
                    for n in op.in_nodes:
                        fill_upstream_color(n, color, is_elmwise, boundary_nodes)
                else:
                    # Add to in nodes of this color
                    in_nodes_list[color].append(node)
                    boundary_nodes.append(node)
            else:
                in_nodes_list[color].append(node)

        queue = list(out_nodes)
        while len(queue) > 0:
            node = queue.pop()

            # Find color of this node and upstream nodes
            is_elmwise = isinstance(node.creator_op, FusionOp)
            color = find_upstream_color(node, is_elmwise)
            if color is None:
                color = n_colors
                n_colors += 1
                is_elmwise_list.append(is_elmwise)
                in_nodes_list.append([])
                out_nodes_list.append([])

            out_nodes_list[color].append(node)

            # Fill upstream node with found color
            boundary_nodes = []
            fill_upstream_color(node, color, is_elmwise, boundary_nodes)

            queue.extend(boundary_nodes)

        assert len(in_nodes_list) == n_colors
        assert len(out_nodes_list) == n_colors
        assert len(is_elmwise_list) == n_colors

        return [(is_elmwise_list[c], in_nodes_list[c], out_nodes_list[c])
                for c in range(n_colors)]

    def __call__(self, *args):
        param_infos = self.param_infos
        gen = self._gen

         # Broadcast
        brod = core.broadcast(*args)

        #-------------------
        # Prepare arguments
        #-------------------
        in_args = list(brod.values)

        # Allocate output args
        out_args = []
        for i, var in enumerate(self._out_vars):
            if var not in self._in_vars:
                out_args.append(core.ndarray(brod.shape, var.dtype))

        # Append indexer
        indexer = _carray.Indexer(brod.shape)
        inout_args = in_args + out_args + [indexer]

        #
        inout_arg_infos = tuple([kernel_core.RuntimeArgInfo_from_arg(_) for _ in inout_args])
        param_list = kernel_core.ParameterList(
            tuple(param_infos),
            inout_arg_infos)

        # Emit elementwise functions
        param_list_expr = param_list.get_entry_function_param_list()
        code = []
        for fuser in reversed(self.fusers):
            fuser.emit(gen, param_list)
            code.append('{class_name}().compute({param_list_expr});'.format(
                class_name=fuser.name,
                param_list_expr=param_list_expr))

        # Kernel entry function
        gen.emit_kernel_entry_function(param_list, '\n'.join(code))
        kern = gen.get_function(())
        kern.linear_launch(indexer.size, inout_args)

        ret = out_args
        if len(ret) == 1:
            ret, = ret
        else:
            ret = tuple(ret)
        return ret


class _FusionReductionCodeGenerator:

    def __init__(self, name, in_nodes, out_nodes):
        assert len(in_nodes) == 1
        assert len(out_nodes) == 1

        in_node, = in_nodes
        out_node, = out_nodes
        op = out_node.creator_op

        assert isinstance(op, ReductionOp), type(op)

        self.name = name
        self.in_node = in_node
        self.out_node = out_node
        self.op = op

    def emit(self, gen, param_list):
        op = self.op
        block_size = 512
        pre_map_expr, reduce_expr, post_map_expr, reduce_type = op.routine[:4]
        type_preamble = ''
        input_expr = ''
        output_expr = ''
        preamble = ''

        gen.emit_reduction_function(
            self.name, param_list,
            block_size, reduce_type, op.identity,
            pre_map_expr, reduce_expr, post_map_expr,
            type_preamble, input_expr, output_expr, preamble)


class _FusionElementwiseCodeGenerator:

    def __init__(self, name, in_nodes, out_nodes):
        assert len(in_nodes) > 0
        assert len(out_nodes) > 0
        self._submitted_nodes = set()
        self._submitted_ops = set()
        self._submitted_vars = set()
        self._var_num = 0
        self._in_nodes = tuple(in_nodes)
        self._out_vars = []
        self._tmp_vars = []
        self._s = six.StringIO()
        self._op_list = []  # [] of (op, [] of vars)
        self._var_name_map = {}  # {} of id(var) to var name
        self.name = name

        self._in_vars = [node.var for node in in_nodes]
        for i, var in enumerate(self._in_vars):
            self.add_var(var)

        self.submit_out_nodes(out_nodes)

    def emit(self, gen, param_list):
        #---------------
        # Generate code
        #---------------
        operation = ''
        preamble = ''

        func_var_list = self._in_vars + self._out_vars
        param_infos = param_list.params
        arg_infos = param_list.arg_infos

        # Variable declaration
        code = []

        # Function parameters
        for var, param_info, arg_info in zip(func_var_list, param_infos, arg_infos):
            code.append(self._get_declaration_from_var(var, param_info, arg_info))

        # Local variables
        for var in self._tmp_vars:
            code.append(self._get_declaration_from_var(var, None, None))

        # Operations
        for op, vars in self._op_list:
            code.append(self._get_operation_code(op))

        # Emit elementwise module
        operation = '\n'.join(code) + '\n'
        gen.emit_elementwise_function(
            self.name, param_list, operation, preamble)

    def _get_declaration_from_var(self, var, param_info, arg_info):
        assert isinstance(var, _FusionVar), type(var)
        var_name = self._var_name_map[id(var)]
        if var.const is not None:
            return 'const {} {} = {};'.format(
                var.ctype,
                var_name,
                _const_to_str(var.const))
        elif param_info is not None:
            return '{}& {} = {}[_ind.get()];'.format(
                var.ctype, var_name, param_info.get_var_name(arg_info))
        else:
            return '{} {};'.format(var.ctype, var_name)

    def _get_operation_code(self, op):
        assert isinstance(op, FusionOp), type(op)
        indent = '  '
        code = []
        code.append('{')

        nin = len(op.in_nodes)
        nodes = op.in_nodes + op.out_nodes

        # Temporary variables
        for i, (node, tmp_name) in enumerate(zip(nodes, op.param_names)):
            var = node.var
            var_name = self._var_name_map[id(var)]
            if i < nin:
                code.append('  {} {} = {};'.format(var.ctype, tmp_name, var_name))
            else:
                code.append('  {} {};\n'.format(var.ctype, tmp_name))

        # Call
        code.append(op.operation + ';')

        # Output variables
        for i, node in enumerate(op.out_nodes):
            var = node.var
            var_name = self._var_name_map[id(var)]
            tmp_name = op.param_names[nin + i]
            code.append('  {} = {};'.format(var_name, tmp_name))

        code.append('}')
        code.append('')
        return '\n'.join(code)

    def submit_op(self, op):
        """Submit elementwise operation"""
        assert isinstance(op, FusionOp), type(op)

        if id(op) in self._submitted_ops:
            return
        self._submitted_ops.add(id(op))

        # Submit upstream nodes
        vars = []
        for node in op.in_nodes:
            var = self.submit_node(node, None)
            vars.append(var)

        # Add elementwise operation
        self._op_list.append((op, vars))

    def submit_out_nodes(self, nodes):
        for i, node in enumerate(nodes):
            self.submit_out_node(node, i)

    def submit_out_node(self, node, i_out):
        """Submits an output node
        """
        assert isinstance(node, _FusionNode)
        assert i_out is None or i_out >= 0
        self.submit_node(node, i_out)

    def submit_node(self, node, i_out):
        """Submits a node
        """
        assert isinstance(node, _FusionNode), type(node)
        assert i_out is None or i_out >= 0
        var = node.var

        if id(node) in self._submitted_nodes:
            var_name = self._var_name_map[id(var)]
            return var, var_name
        self._submitted_nodes.add(id(node))

        try:
            i_in = self._in_vars.index(var)
        except ValueError:
            i_in = None

        if i_in is None:
            # Variable is not in_var
            # Submit parent op
            if node.creator_op is not None:
                self.submit_op(node.creator_op)

        # Submit variable
        var_name = self._var_name_map.get(id(var))
        if var_name is None:
            var_name = self.add_var(var)

        if i_out is not None:
            self._out_vars.append(var)
        elif i_in is None:
            # Neither in nor out
            self._tmp_vars.append(var)

        return var, var_name

    def add_var(self, var):
        var_name = 'v{}'.format(self._var_num)
        self._var_name_map[id(var)] = var_name
        self._var_num += 1
        return var_name


_kind_score = {
    'b': 0,
    'u': 1,
    'i': 1,
    'f': 2,
}

_dtype_to_ctype = {
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

_dtype_list = [numpy.dtype(_) for _ in '?bhilqBHILQefd']


def _const_to_str(val):
    return str(val).lower() if type(val) is bool else str(val)


def _as_ref(arg):
    if isinstance(arg, _FusionRef):
        return arg
    if not isinstance(arg, (int, float, bool)):
        if not hasattr(arg, 'dtype') or arg.dtype not in _dtype_list:
            raise Exception('Unsupported type %s' % arg_type)
    var = _FusionVar(numpy.dtype(type(arg)), const=arg)
    node = _FusionNode(var, None)
    ref = _FusionRef(node)
    return ref


def _convert(f):
    if isinstance(f, core.ufunc):
        return _convert_from_ufunc(f)
    if isinstance(f, core.ElementwiseKernel):
        raise NotImplementedError()
    if isinstance(f, core.simple_reduction_function):
        raise NotImplementedError()
    raise TypeError("Can't convert from %s to FusionOp" % type(f))


def _should_use_min_scalar(in_vars):
    max_array_kind = -2
    max_scalar_kind = -1
    for v in in_vars:
        kind = _kind_score[v.dtype.kind]
        if v.const is None:
            max_array_kind = max(max_array_kind, kind)
        else:
            max_scalar_kind = max(max_scalar_kind, kind)
    return (max_scalar_kind != -1 and
            max_array_kind >= max_scalar_kind)


def _convert_from_ufunc(ufunc):
    nin = ufunc.nin
    nout = ufunc.nout

    def get_mem(args):
        for i in args:
            if type(i) == _FusionRef:
                return i._mem
        raise Exception('number of ndarray arguments must be more than 0')

    def can_cast1(args, ty_ins):
        for i in six.moves.range(nin):
            if args[i].const is None:
                if not numpy.can_cast(args[i].dtype, ty_ins[i]):
                    return False
            else:
                if not numpy.can_cast(args[i].const, ty_ins[i]):
                    return False
        return True

    def can_cast2(args, ty_ins):
        for i in six.moves.range(nin):
            if not numpy.can_cast(args[i].dtype, ty_ins[i]):
                return False
        return True

    def res(*args, **kwargs):
        arg_refs = [_as_ref(_) for _ in args]
        if 'out' in kwargs:
            arg_refs.append(_as_ref(kwargs.pop('out')))
        if kwargs:
            raise TypeError('Wrong arguments %s' % kwargs)
        assert nin <= len(arg_refs) <= nin + nout
        in_refs = arg_refs[:nin]
        out_refs = arg_refs[nin:]
        in_vars = [r.node.var for r in in_refs]
        in_nodes = [r.node for r in in_refs]
        can_cast = can_cast1 if _should_use_min_scalar(in_vars) else can_cast2

        for ty_ins, ty_outs, op in ufunc._ops:
            ty_ins = [numpy.dtype(_) for _ in ty_ins]
            ty_outs = [numpy.dtype(_) for _ in ty_outs]
            if can_cast(in_vars, ty_ins):
                break
        else:
            raise TypeError('Invalid type cast in \'{}\': {} -> {}'.format(
                ufunc.name,
                [_.dtype for _ in in_refs],
                [_.dtype for _ in out_refs]))

        param_names = (['in%d' % i for i in six.moves.range(nin)] +
                       ['out%d' % i for i in six.moves.range(nout)])
        out_nodes = []
        for i in six.moves.range(nout):
            if i >= len(out_refs):
                var = _FusionVar(ty_outs[i])
            elif numpy.can_cast(ty_outs[i], out_refs[i].dtype, 'same_kind'):
                var = out_vars[i]
            else:
                raise TypeError(
                    'output (typecode \'{}\') could not be coerced '
                    'to provided output parameter (typecode \'{}\') '
                    'according to the casting rule '
                    '"same_kind"'.format(
                        ty_outs[i].char, out_vars[i].dtype.char))
            node = _FusionNode(var, None)
            out_nodes.append(node)

        fusion_op = FusionOp(ufunc.name, op, param_names, nin, nout,
                             in_nodes, out_nodes, ty_ins + ty_outs)
        for node in out_nodes:
            node.creator_op = fusion_op

        if len(out_nodes) == 1:
            return _FusionRef(out_nodes[0])
        else:
            return tuple([_FusioinRef(n) for n in out_nodes])
    return res


def _gather_submodules(ops):
    return {(op.name, tuple(op.types)): op for op in ops}


def _get_pre_code(in_vars, out_vars, operation):
    """Generate preprocess code for reduction"""

    in_params = ', '.join('%s v%s' % (_dtype_to_ctype[v.dtype], v.num)
                          for v in in_vars)
    out_params = ''.join('%s v%s;\n' % (_dtype_to_ctype[v.dtype], v.num)
                         for v in out_vars)
    module_code = string.Template('''
    __device__ ${return_type} _pre_map(${in_params}) {
      ${out_params}
      ${operation};
      return ${return_var};
    }
    ''').substitute(
        return_type=_dtype_to_ctype[out_vars[0].dtype],
        in_params=in_params,
        out_params=out_params,
        operation=operation,
        return_var='v%d' % out_vars[0].num)
    return module_code


def _get_reduce_op(ops, dtype):
    """Generate reduction operation code"""
    for i in ops._ops:
        if numpy.can_cast(dtype.type, i[0][0]):
            return i
    raise TypeError("Type is mismatched. %s(...), %s" % (ops.name, dtype.type))


def _get_post_code(post_vars, operation, post_out):
    """Generate postprocessing code for reduction"""

    module_code = string.Template('''
    __device__ ${return_type} _post_map(${arg_type} v0) {
      ${operation};
      return v${return_var};
    }
    ''').substitute(
        arg_type=_dtype_to_ctype[post_vars[0].dtype],
        return_type=_dtype_to_ctype[post_vars[post_out.num].dtype],
        operation=operation,
        return_var=post_out.num)
    return module_code


def _get_fix_code(data_type, fixed_type, operation):
    """Generate fix code for reduction"""

    module_code = string.Template('''
    __device__ ${fixed_type} _post_fix(${data_type} a) {
      ${fixed_type} out0;
      ${operation};
      return out0;
    }
    ''').substitute(
        data_type=data_type,
        fixed_type=_dtype_to_ctype[fixed_type],
        operation=operation)
    return module_code


# Temporary implementation
def _apply_reduction(reduce, ref):
    pre_type = ref.dtype
    (in_type,), (out_type,), routines  = _get_reduce_op(reduce._raw, pre_type)
    reduce_type = numpy.dtype(out_type)
    #rtype = rt[3]
    #post_type = "type_in0_raw" if rtype is None else rtype
    #pre_code += "typedef %s type_in0_raw;\n" % _dtype_to_ctype[reduce_type]

    var = _FusionVar(reduce_type, False)
    node =_FusionNode(var, None)

    op = ReductionOp(
        'reduce', routines, ['in1', 'out1'],
        ref.node, node, [pre_type, reduce_type], reduce.identity)
    node.creator_op = op
    return _FusionRef(node)


def _get_fusion(func, args, nin, reduce, post_map, identity, input_types, name=None):
    in_vars = [_FusionVar(dtype) for dtype in input_types]
    in_refs = [_FusionRef(_FusionNode(v, None)) for v in in_vars]
    out_refs = func(*in_refs)
    out_refs = list(out_refs) if isinstance(out_refs, tuple) else [out_refs]
    out_refs = [_ for _ in out_refs if _ is not None]  # Do we need this?

    if name is None:
        name = 'fusion'

    if reduce is not None:
        out_refs_ = []
        out_refs = [_apply_reduction(reduce, ref) for ref in out_refs]

    gen = _FusionCodeGenerator(name, in_refs, out_refs)
    return gen


    if reduce is None:
        return gen
    else:
        if nout != 1:
            raise Exception("Wrong number of number of arguments")
        # pre-map
        pre_type = out_vars[0].dtype
        pre_code = _get_pre_code(in_vars, out_vars, operation)

        # reduce
        reduce_op = _get_reduce_op(reduce._raw, pre_type)
        reduce_code = reduce_op[2][1]
        reduce_type = numpy.dtype(reduce_op[1][0])
        rtype = reduce_op[2][3]
        post_type = "type_in0_raw" if rtype is None else rtype
        pre_code += "typedef %s type_in0_raw;\n" % _dtype_to_ctype[reduce_type]

        # post-map
        post_in = [_FusionVar(0, reduce_type)]
        mem = _FusionMem(post_in)
        post_in_ref = [_FusionRef(_, mem) for _ in post_in]
        post_out = _normalize_arg(post_map(*post_in_ref), mem)
        if type(post_out) == tuple:
            raise Exception("Can't reduce a tuple")
        post_vars = mem.var_list
        post_ops = mem.op_list
        post_code = ''.join(_get_declaration_from_var(_)
                            for _ in post_vars[1:])
        post_code += ''.join(_get_declaration_from_op(_) for _ in post_ops)
        post_code += '\n'.join(_get_operation_code(_) for _ in post_ops)
        post_code = _get_post_code(post_vars, post_code, post_out)
        post_code += _get_fix_code(post_type, reduce_type, reduce_op[2][2])

        submodules = _gather_submodules(op_list + post_ops)
        submodule_code = ''.join(_get_submodule_code(v)
                                 for v in submodules.values())
        submodule_code += reduce._raw._preamble + pre_code + post_code
        operation_args = ['v' + str(i) for i in six.moves.range(nin)]
        operation = '_pre_map(' + ', '.join(operation_args) + ')'
        out_params = '%s res' % post_out.dtype
        print('--- in_params')
        print(in_params)
        print('--- out_params')
        print(out_params)
        print('--- operation')
        print(operation)
        print('--- reduce_code')
        print(reduce_code)
        print('--- preamble (submodule_code)')
        print(submodule_code)
        print('---')
        return core.ReductionKernel(in_params, out_params, operation,
                                    reduce_code,
                                    'res = _post_map(_post_fix(a))',
                                    identity,
                                    reduce_type=post_type,
                                    preamble=submodule_code)


class Fusion(object):

    """Function class.

    This class can be get by using `fuse` function and
    works like `ElementwiseKernel` or `ReductionKernel`.

    Attributes:
        func (function): The function before fusing.
        name (str): The name of the function.
        reduce (ufunc): Reduction ufunc.
        post_map (function): Mapping function for reduced values.
    """

    def __init__(self, func, input_num, reduce, post_map):
        self.func = func
        self.name = func.__name__
        self.input_num = input_num
        self.reduce = reduce
        self.post_map = post_map
        self.identity = None if reduce is None else self.reduce._raw.identity
        self._memo = {}

    def __repr__(self):
        return "<Fusion '%s'>" % self.name

    def __call__(self, *args, **kwargs):
        _thread_local.in_fusion = True
        try:
            return self._call(*args, **kwargs)
        finally:
            _thread_local.in_fusion = False

    def _call(self, *args, **kwargs):
        axis = kwargs['axis'] if 'axis' in kwargs else None
        if len(args) == 0:
            raise Exception('number of arguments must be more than 0')
        if builtins.any(
                not isinstance(_, (core.ndarray, numpy.ndarray, numpy.generic))
                for _ in args):
            raise TypeError('Invalid argument type for \'{}\': ({})'.format(
                self.name,
                ', '.join(repr(type(_)) for _ in args)))

        def is_cupy_data(a):
            return isinstance(a, (core.ndarray, numpy.generic))
        if builtins.all(is_cupy_data(_) for _ in args):
            types = [_.dtype for _ in args]
            key = tuple(types)
            if key not in self._memo:
                if self.input_num is not None:
                    nin = self.input_num
                else:
                    nin = len(args)
                f = _get_fusion(self.func, args, nin, self.reduce,
                                self.post_map, self.identity, types)
                self._memo[key] = f
            f = self._memo[key]
            if self.reduce is None:
                return f(*args)
            else:
                #return f(*args, axis=axis)
                return f(*args)
        else:
            if builtins.any(type(_) is core.ndarray for _ in args):
                types = '.'.join(repr(type(_)) for _ in args)
                message = "Can't fuse \n %s(%s)" % (self.name, types)
                warnings.warn(message)
            if self.reduce is None:
                return self.func(*args)
            elif axis is None:
                return self.post_map(self.reduce(self.func(*args)))
            else:
                return self.post_map(self.reduce(self.func(*args), axis=axis))


def fuse(*args, **kwargs):
    """Function fusing decorator.

    This decorator can be used to define an elementwise or reduction kernel
    more easily than `ElementwiseKernel` class or `ReductionKernel` class.

    This decorator makes `Fusion` class from the given function.

    Args:
        input_num (int): Number of input arguments of the given function.
        reduce (function): The reduce function which is applied after
            pre-mapping step. If not assigned, reduction step is skipped.
        post_map (function): Mapping function for reduced values.
            If not assigned, post_map step is skipped.
    """
    util.experimental('cupy.core.fusion')

    def wrapper(f, input_num=None, reduce=None, post_map=lambda x: x):
        return Fusion(f, input_num, reduce, post_map)

    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return functools.update_wrapper(wrapper(args[0]), args[0])
    else:
        return lambda f: functools.update_wrapper(
            wrapper(f, *args, **kwargs), f)


def build_kernel_name(entity):
    if isinstance(entity, (FusionOp, ReductionOp)):
        return entity.build_kernel_name()
    elif isinstance(entity, _FusionVar):
        return entity.build_kernel_name()
    elif isinstance(entity, _FusionRef):
        return entity.build_kernel_name()
    else:
        assert False, type(entity)


class ufunc(core.ufunc):

    def __init__(self, fusion_op, cupy_op, numpy_op):
        self.name = fusion_op.name
        self.nin = fusion_op.nin
        self.nout = fusion_op.nout
        self.nargs = fusion_op.nargs
        self.__doc__ = fusion_op.__doc__

        self._fusion_op = fusion_op
        self._cupy_op = cupy_op
        self._numpy_op = numpy_op

    def __repr__(self):
        return repr(self._cupy_op)

    def __call__(self, *args, **kwargs):
        in_fusion = getattr(_thread_local, 'in_fusion', False)
        if in_fusion:
            if builtins.any(isinstance(_, _FusionRef) for _ in args):
                return _convert(self._fusion_op)(*args, **kwargs)
            elif builtins.any(isinstance(_, numpy.ndarray) for _ in args):
                return self._numpy_op(*args, **kwargs)

        return self._cupy_op(*args, **kwargs)

    __doc__ = core.ufunc.__doc__
    __call__.__doc__ = core.ufunc.__call__.__doc__


def _create_ufunc(cupy_ufunc, numpy_ufunc):
    return ufunc(cupy_ufunc, cupy_ufunc, numpy_ufunc)


_where = ufunc(sorting.search._where_ufunc,
               sorting.search.where, numpy.where)

_clip = ufunc(core._clip, math.misc.clip, numpy.clip)

_elementwise_copy = ufunc(core._elementwise_copy,
                          creation.from_data.copy, numpy.copy)


def where(*args, **kwargs):
    return _where(*args, **kwargs)


def clip(*args, **kwargs):
    return _clip(*args, **kwargs)


def copy(*args, **kwargs):
    return _elementwise_copy(*args, **kwargs)


bitwise_and = _create_ufunc(core.bitwise_and, numpy.bitwise_and)
bitwise_or = _create_ufunc(core.bitwise_or, numpy.bitwise_or)
bitwise_xor = _create_ufunc(core.bitwise_xor, numpy.bitwise_xor)
invert = _create_ufunc(core.invert, numpy.invert)
left_shift = _create_ufunc(core.left_shift, numpy.left_shift)
right_shift = _create_ufunc(core.right_shift, numpy.right_shift)

greater = _create_ufunc(core.greater, numpy.greater)
greater_equal = _create_ufunc(core.greater_equal, numpy.greater_equal)
less = _create_ufunc(core.less, numpy.less)
less_equal = _create_ufunc(core.less_equal, numpy.less_equal)
equal = _create_ufunc(core.equal, numpy.equal)
not_equal = _create_ufunc(core.not_equal, numpy.not_equal)

isfinite = _create_ufunc(logic.content.isfinite, numpy.isfinite)
isinf = _create_ufunc(logic.content.isinf, numpy.isinf)
isnan = _create_ufunc(logic.content.isnan, numpy.isnan)

logical_and = _create_ufunc(logic.ops.logical_and, numpy.logical_and)
logical_or = _create_ufunc(logic.ops.logical_or, numpy.logical_or)
logical_not = _create_ufunc(logic.ops.logical_not, numpy.logical_not)
logical_xor = _create_ufunc(logic.ops.logical_xor, numpy.logical_xor)

sin = _create_ufunc(math.trigonometric.sin, numpy.sin)
cos = _create_ufunc(math.trigonometric.cos, numpy.cos)
tan = _create_ufunc(math.trigonometric.tan, numpy.tan)
arcsin = _create_ufunc(math.trigonometric.arcsin, numpy.arcsin)
arccos = _create_ufunc(math.trigonometric.arccos, numpy.arccos)
arctan = _create_ufunc(math.trigonometric.arctan, numpy.arctan)
arctan2 = _create_ufunc(math.trigonometric.arctan2, numpy.arctan2)
hypot = _create_ufunc(math.trigonometric.hypot, numpy.hypot)
deg2rad = _create_ufunc(math.trigonometric.deg2rad, numpy.deg2rad)
rad2deg = _create_ufunc(math.trigonometric.rad2deg, numpy.rad2deg)
degrees = _create_ufunc(math.trigonometric.degrees, numpy.degrees)
radians = _create_ufunc(math.trigonometric.radians, numpy.radians)

sinh = _create_ufunc(math.hyperbolic.sinh, numpy.sinh)
cosh = _create_ufunc(math.hyperbolic.cosh, numpy.cosh)
tanh = _create_ufunc(math.hyperbolic.tanh, numpy.tanh)
arcsinh = _create_ufunc(math.hyperbolic.arcsinh, numpy.arcsinh)
arccosh = _create_ufunc(math.hyperbolic.arccosh, numpy.arccosh)
arctanh = _create_ufunc(math.hyperbolic.arctanh, numpy.arctanh)

rint = _create_ufunc(math.rounding.rint, numpy.rint)
floor = _create_ufunc(math.rounding.floor, numpy.floor)
ceil = _create_ufunc(math.rounding.ceil, numpy.ceil)
trunc = _create_ufunc(math.rounding.trunc, numpy.trunc)
fix = _create_ufunc(math.rounding.fix, numpy.fix)

exp = _create_ufunc(math.explog.exp, numpy.exp)
expm1 = _create_ufunc(math.explog.expm1, numpy.expm1)
exp2 = _create_ufunc(math.explog.exp2, numpy.exp2)
log = _create_ufunc(math.explog.log, numpy.log)
log10 = _create_ufunc(math.explog.log10, numpy.log10)
log2 = _create_ufunc(math.explog.log2, numpy.log2)
log1p = _create_ufunc(math.explog.log1p, numpy.log1p)
logaddexp = _create_ufunc(math.explog.logaddexp, numpy.logaddexp)
logaddexp2 = _create_ufunc(math.explog.logaddexp2, numpy.logaddexp2)

signbit = _create_ufunc(math.floating.signbit, numpy.signbit)
copysign = _create_ufunc(math.floating.copysign, numpy.copysign)
ldexp = _create_ufunc(math.floating.ldexp, numpy.ldexp)
frexp = _create_ufunc(math.floating.frexp, numpy.frexp)
nextafter = _create_ufunc(math.floating.nextafter, numpy.nextafter)

add = _create_ufunc(math.arithmetic.add, numpy.add)
reciprocal = _create_ufunc(math.arithmetic.reciprocal, numpy.reciprocal)
negative = _create_ufunc(math.arithmetic.negative, numpy.negative)
multiply = _create_ufunc(math.arithmetic.multiply, numpy.multiply)
divide = _create_ufunc(math.arithmetic.divide, numpy.divide)
power = _create_ufunc(math.arithmetic.power, numpy.power)
subtract = _create_ufunc(math.arithmetic.subtract, numpy.subtract)
true_divide = _create_ufunc(math.arithmetic.true_divide, numpy.true_divide)
floor_divide = _create_ufunc(math.arithmetic.floor_divide, numpy.floor_divide)
fmod = _create_ufunc(math.arithmetic.fmod, numpy.fmod)
mod = _create_ufunc(math.arithmetic.remainder, numpy.mod)
modf = _create_ufunc(math.arithmetic.modf, numpy.modf)
remainder = _create_ufunc(math.arithmetic.remainder, numpy.remainder)

sqrt = _create_ufunc(math.misc.sqrt, numpy.sqrt)
sqrt_fixed = _create_ufunc(math.misc.sqrt_fixed, numpy.sqrt)
square = _create_ufunc(math.misc.square, numpy.square)
absolute = _create_ufunc(math.misc.absolute, numpy.absolute)
abs = _create_ufunc(math.misc.absolute, numpy.abs)
sign = _create_ufunc(math.misc.sign, numpy.sign)
maximum = _create_ufunc(math.misc.maximum, numpy.maximum)
minimum = _create_ufunc(math.misc.minimum, numpy.minimum)
fmax = _create_ufunc(math.misc.fmax, numpy.fmax)
fmin = _create_ufunc(math.misc.fmin, numpy.fmin)


class reduction(object):

    def __init__(self, fusion_op, cupy_op, numpy_op):
        self._fusion_op = fusion_op
        self._cupy_op = cupy_op
        self._numpy_op = numpy_op

        if numpy_op == numpy.sum:
            self.identity = cupy_op.identity

    def __call__(self, *args, **kwargs):
        in_fusion = getattr(_thread_local, 'in_fusion', False)
        if in_fusion:
            if builtins.any(isinstance(_, _FusionRef) for _ in args):
                return _convert(self._fusion_op)(*args, **kwargs)
            elif builtins.any(isinstance(_, numpy.ndarray) for _ in args):
                return self._numpy_op(*args, **kwargs)

        return self._cupy_op(*args, **kwargs)


_all = reduction(core._all, core._all, numpy.all)
_any = reduction(core._any, core._any, numpy.any)
_sum = reduction(core._sum, core._sum, numpy.sum)
_prod = reduction(core._prod, core._prod, numpy.prod)
_amax = reduction(core._amax, core._amax, numpy.amax)
_amin = reduction(core._amin, core._amin, numpy.amin)


all = _all
any = _any
sum = _sum
prod = _prod
amax = _amax
amin = _amin
