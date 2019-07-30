from utensor_cgen.ir import OperationInfo, uTensorGraph
from utensor_cgen.logger import logger
from graphviz import Digraph
import re

def sanitize_name(n):
    return re.sub("[^_a-zA-Z0-9]", "_", n)

def viz_graph(out_fname, view, ugraph):
    dot = Digraph()
    nodes = {}
    i = 0
    for node in ugraph.ops:
        dot.attr('node', shape='ellipse')
        nodes[node.name] = chr(ord('a') + i)
        dot.node(nodes[node.name], sanitize_name("%s: %s" % (node.name, node.op_type)))
        i += 1
        for n in node.input_tensors:
            if n.name in nodes:
                continue
            nodes[n.name] = chr(ord('a') + i)
            dot.attr('node', shape='box')
            dot.node(nodes[n.name], sanitize_name("%s: Tensor" % n.name))
            i += 1
        for n in node.output_tensors:
            if n.name in nodes:
                continue
            nodes[n.name] = chr(ord('a') + i)
            dot.attr('node', shape='box')
            dot.node(nodes[n.name], sanitize_name("%s: Tensor" % n.name))
            i += 1
    for node in ugraph.ops:
        for n in node.input_tensors:
            dot.edge(nodes[n.name], nodes[node.name])
        for n in node.output_tensors:
            dot.edge(nodes[node.name], nodes[n.name])
    dot.render(out_fname, view=view)
    logger.info('graph visualization file generated: %s', out_fname)
