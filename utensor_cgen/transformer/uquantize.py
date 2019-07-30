from tensorflow.tools.graph_transforms import TransformGraph

from utensor_cgen.ir.base import TensorInfo, OperationInfo, uTensorGraph
from utensor_cgen.frontend.tensorflow import GraphDefParser
from utensor_cgen.utils import topologic_order_graph
import copy

from .base import Transformer

import numpy as np
import re

from graphviz import Digraph

__all__ = ['UQuantizeTransformer']

__Quantizable__ = ["MaxPool",
                   "MatMul",
                   "Relu",
                   "Add",
                   "Mul",
                   "Reshape",
                   "FakeQuantWithMinMaxVars",
                   "FusedConv2DMaxPool",
                   "Conv2D"]

__DifferentOrder__ = ["Reshape", "Conv2D"]

def get_quantize_params2(minv, maxv):
    number_of_bits = 8
    number_of_steps = np.uint64(1 << number_of_bits)
    range_min = minv
    if maxv == minv:
        range_scale = 0.0
    else:
        range_scale = (number_of_steps - 1.0)/(maxv - minv)
    range_min_scaled = round(range_min*range_scale)
    return range_min, range_scale, range_min_scaled

def lower_bound_float():
    #f = np.finfo(np.float)
    return max(0.0, -2.147483648e+09)
def upper_bound_float():
    return min(255.0, 2.147483520e+09)
def lowest_quantized():
    return 0.0

def quantize_floats(x):
    input_min_range = min(x)
    input_max_range = max(x)
    min_range = min(0.0, input_min_range)
    epsilon = max(1.0, max(abs(input_min_range), abs(input_max_range))) / 100.0
    max_range = max(input_max_range, min_range + epsilon)
    max_range = max(0.0, max_range)
    range_min, range_scale, range_min_scaled = get_quantize_params2(min_range, max_range)
    qx = []
    for xi in x:
        val = round(xi * range_scale)
        val -= range_min_scaled - lowest_quantized()
        val = max(val, lower_bound_float())
        val = min(val, upper_bound_float())
        inttmp = np.uint32(val)
        qx.append(np.uint8(inttmp))
    #print(qx, min_range, max_range)
    return qx, min_range, max_range


def op_is_quantizable_const(op):
    if op.op_type == 'Const' and op.op_attr['value'].value.dtype == np.float32 and op.op_attr['value'].value.np_array.size > 1:
    #if op.op_type == 'Const'  and op.op_attr['value'].value.np_array.size > 1:
        return True
    else:
        return False

def swap(mlist, i1, i2):
    mlist[i1], mlist[i2] = mlist[i2], mlist[i1]
    return mlist

def sanitize_name(n):
    return re.sub("[^_a-zA-Z0-9]", "_", n)

def add_op_to_viz(g, op):
    g.attr('node', shape='ellipse')
    g.node(sanitize_name(op.name), label="%s\n%s" % (op.op_type, sanitize_name(op.name)))
    g.attr('node', shape='box')
    for input_tensor in op.input_tensors:
        g.edge(sanitize_name(input_tensor.name), sanitize_name(op.name), len="1.0")
    for output_tensor in op.output_tensors:
        g.node(sanitize_name(output_tensor.name))
        g.edge(sanitize_name(op.name), sanitize_name(output_tensor.name),  len="1.0")

class UQuantizeTransformer(Transformer):

  METHOD_NAME = 'uquantize'
  KWARGS_NAMESCOPE = '_quantize'

  def transform(self, ugraph):
    graph_def = ugraph.graph_def
    mugraph = uTensorGraph(output_nodes=ugraph.output_nodes,
                           backend="tensorflow")
    quantized_tensors = []
    g = Digraph('ER', filename="uquantize.gv")

    #import pdb; pdb.set_trace()
    op_attr_const = None
    for op in ugraph.ops:
        if op.op_type == "Const":
            op_attr_const = copy.deepcopy(op.op_attr)
            break

    import pdb; pdb.set_trace()
    for op in ugraph.ops:
        #print(op.op_type, op.name, op.op_attr)
        if op.op_type == "Placeholder":
            # Use placeholder original tensor name as the output of QuantizeV2
            # update the placeholder output tensor.name to temp name
            input_tensor = op.output_tensors[0]
            old_name = input_tensor.name
            input_tensor.ugraph = mugraph
            input_tensor.name = old_name + "_pq"
            qop_name = "%s_q" % op.name
            opP = OperationInfo(name=op.name,
                                input_tensors=op.input_tensors, # Empty so who cares
                                output_tensors=[input_tensor], # Placeholder output == quantize input
                                op_type=op.op_type,
                                backend="tensorflow",
                                op_attr=op.op_attr,
                                ugraph=mugraph)
            add_op_to_viz(g, opP)
            # Add min and max ops
            resTensorDimConst = TensorInfo(name="%s_reshape_dim" % input_tensor.name,
                                   ugraph=mugraph,
                                   op_name="%s_reshape_const" % op.name,
                                   dtype=np.dtype(np.int32),
                                   shape=[1])
            op_attr = copy.deepcopy(op_attr_const)
            op_attr['value'].value.np_array= np.array([-1], dtype=np.int32)
            op_attr['value'].value.dtype = np.dtype(np.int32)
            op_red_dims = OperationInfo(name="%s_reshape_const" % op.name,
                                        input_tensors=[],
                                        output_tensors=[resTensorDimConst],
                                        op_type="Const",
                                        backend="tensorflow",
                                        op_attr=op_attr,
                                        ugraph=mugraph)
            add_op_to_viz(g, op_red_dims)
            redTensorDim = TensorInfo(name="%s_red_dim" % input_tensor.name,
                                   ugraph=mugraph,
                                   op_name="%s_red" % op.name,
                                   dtype=np.dtype(np.uint32),
                                   shape=[1])
            op_attr = copy.deepcopy(op_attr_const)
            op_attr['value'].value.np_array= np.array([0], dtype=np.uint32)
            op_attr['value'].value.dtype = np.dtype(np.uint32)
            op_red_dims = OperationInfo(name="%s_red" % op.name,
                                        input_tensors=[],
                                        output_tensors=[redTensorDim],
                                        op_type="Const",
                                        backend="tensorflow",
                                        op_attr=op_attr,
                                        ugraph=mugraph)
            add_op_to_viz(g, op_red_dims)
            
            reshapeTensor = TensorInfo(name="%s_reshape" % input_tensor.name,
                                   ugraph=mugraph,
                                   op_name="%s_reshape" % op.name,
                                   dtype=input_tensor.dtype,
                                   shape=input_tensor.shape) # TODO Check this 
            op_reshape = OperationInfo(name="%s_reshape" % op.name,
                                        input_tensors=[input_tensor, resTensorDimConst],
                                        output_tensors=[reshapeTensor],
                                        op_type="Reshape",
                                        backend="tensorflow",
                                        op_attr=op.op_attr,
                                        ugraph=mugraph)
            add_op_to_viz(g, op_reshape)

            minTensor = TensorInfo(name="%s_q_min" % old_name,
                                   ugraph=mugraph,
                                   op_name="%s_min" % op.name,
                                   dtype=np.dtype(np.float32),
                                   shape=[1])
            maxTensor = TensorInfo(name="%s_q_max" % old_name,
                                   ugraph=mugraph,
                                   op_name="%s_max" % op.name,
                                   dtype=np.dtype(np.float32),
                                   shape=[1])
            opMin = OperationInfo(name="%s_min" % op.name,
                                input_tensors=[reshapeTensor, redTensorDim],
                                output_tensors=[minTensor], 
                                op_type="Min",
                                backend="tensorflow",
                                op_attr=op.op_attr,
                                ugraph=mugraph)
            opMax = OperationInfo(name="%s_max" % op.name,
                                input_tensors=[reshapeTensor, redTensorDim],
                                output_tensors=[maxTensor], 
                                op_type="Max",
                                backend="tensorflow",
                                op_attr=op.op_attr,
                                ugraph=mugraph)
            add_op_to_viz(g, opMin)
            add_op_to_viz(g, opMax)

            outTensor_qnt = TensorInfo(name=old_name,
                                       ugraph=mugraph,
                                       op_name=qop_name,
                                       dtype=np.dtype(np.uint8),
                                       shape=input_tensor.shape)
            outTensor_min = TensorInfo(name="%s_min" % old_name,
                                   ugraph=mugraph,
                                   op_name=qop_name,
                                   dtype=np.dtype(np.float32),
                                   shape=[1])
            outTensor_max = TensorInfo(name="%s_max" % old_name,
                                   ugraph=mugraph,
                                   op_name=qop_name,
                                   dtype=np.dtype(np.float32),
                                   shape=[1])
            op_attr = copy.deepcopy(op.op_attr)
            op_attr['dtype'] = np.dtype(np.uint8)
            #op_attr['dtype'] = 'uint8'
            qop = OperationInfo(name=qop_name,
                                     input_tensors=[input_tensor, minTensor, maxTensor],
                                     output_tensors=[outTensor_qnt, outTensor_min, outTensor_max],
                                     op_type="QuantizeV2",
                                     backend="tensorflow",
                                     op_attr=op_attr,
                                     ugraph=mugraph)
            add_op_to_viz(g, qop)



        elif op_is_quantizable_const(op):
            #import pdb; pdb.set_trace()
            data = op.op_attr['value'].value.np_array
            origshape = data.shape
            qx, min_range, max_range = quantize_floats(data.flatten())
            qx = np.reshape(np.array(qx, dtype=np.dtype(np.uint8)), origshape)
            op_attr = copy.deepcopy(op.op_attr)
            op_attr['value'].value.np_array = qx
            op_attr['value'].value.dtype = np.dtype(np.uint8)
            #op_attr['value'].value.dtype = 'uint8'
            output_tensors = op.output_tensors
            quantized_tensors.append(op.output_tensors[0].name)
            # Constants only have one output tensor
            for output_tensor in output_tensors:
                output_tensor.ugraph=mugraph
                #output_tensor.name += "_t"
            input_tensors = op.input_tensors
            for input_tensor in input_tensors:
                input_tensor.ugraph=mugraph

            qop_info = OperationInfo(name=op.name,
                                     input_tensors=input_tensors,
                                     output_tensors=output_tensors,
                                     op_type=op.op_type,
                                     backend="tensorflow",
                                     op_attr=op_attr,
                                     ugraph=mugraph)
            
            add_op_to_viz(g, qop_info)

            #mugraph.ops_info[qop_info.name] = qop_info
            # Push min and max ranges TODO Validate this over min()/max()
            op_attr = copy.deepcopy(op.op_attr)
            op_attr['value'].value.np_array = np.array([min_range], dtype=np.float32)
            op_attr['value'].value.dtype = np.dtype(np.float32)
            tshape = op_attr['value'].value.np_array.shape
            outTensor = TensorInfo(name="%s_min" % op.output_tensors[0].name,
                                   ugraph=mugraph,
                                   op_name="%s_min" % op.name,
                                   dtype=np.dtype(np.float32),
                                   shape=list(tshape))
            qop_info_min = OperationInfo(name="%s_min" % op.name,
                                     input_tensors=[],
                                     output_tensors=[outTensor],
                                     op_type=op.op_type,
                                     backend="tensorflow",
                                     op_attr=op_attr,
                                     ugraph=mugraph)
            add_op_to_viz(g, qop_info_min)
            #mugraph.add_op(qop_info_min, sort=False)
            #mugraph.ops_info[qop_info_min.name] = qop_info_min
            op_attr = copy.deepcopy(op.op_attr)
            op_attr['value'].value.np_array = np.array([max_range], dtype=np.float32)
            op_attr['value'].value.dtype = np.dtype(np.float32)
            tshape = op_attr['value'].value.np_array.shape
            outTensor = TensorInfo(name="%s_max" % op.output_tensors[0].name,
                                   ugraph=mugraph,
                                   op_name="%s_max" % op.name,
                                   dtype=np.dtype(np.float32),
                                   shape=list(tshape))
            qop_info_max = OperationInfo(name="%s_max" % op.name,
                                     input_tensors=[],
                                     output_tensors=[outTensor],
                                     op_type=op.op_type,
                                     backend="tensorflow",
                                     op_attr=op_attr,
                                     ugraph=mugraph)
            add_op_to_viz(g, qop_info_max)
            #mugraph.add_op(qop_info_max, sort=False)
            #mugraph.ops_info[qop_info_max.name] = qop_info_max
        elif op.op_type in __Quantizable__:
            ot = op.output_tensors
            it = op.input_tensors
            input_tensors = []
            output_tensors = []
            
            qual_list =  ["", "_min", "_max"]
            if op.op_type == "FakeQuantWithMinMaxVars":
                qual_list = [""]
            for output_tensor in ot:
                for qual in ["", "_min", "_max"]:
                    if qual == "":
                        dtype = output_tensor.dtype
                        shape = output_tensor.shape
                    else:
                        dtype = np.dtype(np.float32)
                        shape = [1]

                    quantized_tensors.append(output_tensor.name)
                    output_tensors.append(TensorInfo(name="%s%s" % (output_tensor.name, qual),
                                                     ugraph=mugraph,
                                                     op_name="%s" % op.name,
                                                     dtype=dtype,
                                                     shape=shape))
            for input_tensor in it:
                # TODO add check here to make sure quantized inputs exist
                itensor_name = input_tensor.name
                for qual in qual_list:
                    op_name = input_tensor.op.name
                    if qual == "":
                        dtype = input_tensor.dtype
                        shape = input_tensor.shape
                    else: # min or max
                        dtype = np.dtype(np.float32)
                        shape = [1]
                        if input_tensor.op.op_type == "GatherV2": # TODO FCUKING HACK
                            itensor_name = input_tensor.op.input_tensors[0].name

                    if input_tensor.op.op_type == "Const":
                        op_name = "%s%s" % ( input_tensor.op.name, qual)
                    if input_tensor.op.op_type == "Placeholder":
                        op_name = "%s_q" % input_tensor.op.name ## update placeholder to point at QuantizeV2 opname

                    input_tensors.append(TensorInfo(name="%s%s" % (itensor_name, qual),
                                                     ugraph=mugraph,
                                                     op_name=op_name,
                                                     dtype=dtype,
                                                     shape=shape))
                    if input_tensor.op.op_type == "Const" and not op_is_quantizable_const(input_tensor.op):
                        break

            if op.op_type in __DifferentOrder__:
                if op.op_type == "Reshape":
                    # x0 y0 x1 x2
                    swap(input_tensors, 1, 3)
                    swap(input_tensors, 2, 3)
                elif op.op_type == "Conv2D":
                    swap(input_tensors, 1, 3)
                    swap(input_tensors, 2, 3)

            if op.op_type == "FakeQuantWithMinMaxVars":
                op_type = "QuantizeV2"
            else:
                op_type = "Quantized%s" % op.op_type
            op_attr = copy.deepcopy(op.op_attr)
            qop = OperationInfo(name=op.name,
                                input_tensors=input_tensors,
                                output_tensors=output_tensors,
                                op_type=op_type,
                                backend="tensorflow",
                                op_attr=op_attr,
                                ugraph=mugraph)
            add_op_to_viz(g, qop)
            #print(qop.op_type, qop.name, [i.name for i in qop.input_tensors], [i.name for i in qop.output_tensors])


        else:
            #mugraph.add_op(op)
            #mugraph.ops_info[op.name] = op
            #mugraph.ops_info[op.name].ugraph = mugraph
            output_tensors = op.output_tensors
            for output_tensor in output_tensors:
                output_tensor.ugraph=mugraph
            input_tensors = op.input_tensors
            for input_tensor in input_tensors:
                #if input_tensor.name in quantized_tensors:
                #    input_tensor.name += "_t"
                input_tensor.ugraph=mugraph
            opC = OperationInfo(name=op.name,
                                input_tensors=input_tensors,
                                output_tensors=output_tensors,
                                op_type=op.op_type,
                                backend="tensorflow",
                                op_attr=op.op_attr,
                                ugraph=mugraph)
            add_op_to_viz(g, opC)

    import pdb; pdb.set_trace()
    #from utensor_cgen.ir.misc.graph_viz import viz_graph
    topologic_order_graph(mugraph)
    #viz_graph('out_graph', True, mugraph)
    #import pdb; pdb.set_trace()

    #quant_graph_def = TransformGraph(input_graph_def=graph_def,
    #                                 inputs=[],
    #                                 outputs=ugraph.output_nodes,
    #                                 transforms=["quantize_weights", "quantize_nodes"])
    #return GraphDefParser.parse(quant_graph_def,
    #                            output_nodes=ugraph.output_nodes)
    return mugraph
