import tensorflow as tf
from lib.proto import mace_pb2
from collections import OrderedDict

def sort_tf_node(node, nodes_map, ordered_nodes_map):
    if node.name not in ordered_nodes_map:
        for input_tensor_name in node.input:
            input_node_name = input_tensor_name.split(':')[
                0] if ':' in input_tensor_name else input_tensor_name
            if input_node_name not in nodes_map or input_node_name in ordered_nodes_map:
                continue

            input_node = nodes_map[input_node_name]
            sort_tf_node(input_node, nodes_map, ordered_nodes_map)
        ordered_nodes_map[node.name] = node

def sort_tf_graph(graph_def):
    nodes_map = {}
    ordered_nodes_map = OrderedDict()
    for node in graph_def.node:
        nodes_map[node.name] = node
    for node in graph_def.node:
        sort_tf_node(node, nodes_map, ordered_nodes_map)
    sorted_graph = tf.GraphDef()
    sorted_graph.node.extend([node for node in ordered_nodes_map.values()])
    return sorted_graph


def sort_mace_node(node, nodes_map, ordered_nodes_map):
    if node.name not in ordered_nodes_map:
        for input_tensor_name in node.input:
            input_node_name = input_tensor_name.split(':')[
                0] if ':' in input_tensor_name else input_tensor_name
            if input_node_name not in nodes_map or input_node_name in ordered_nodes_map:
                continue

            input_node = nodes_map[input_node_name]
            sort_mace_node(input_node, nodes_map, ordered_nodes_map)
        ordered_nodes_map[node.name] = node

def sort_mace_graph(graph_def, output_name):
    nodes_map = {}
    ordered_nodes_map = OrderedDict()
    for node in graph_def.op:
        nodes_map[node.name] = node
    sort_mace_node(nodes_map[output_name], nodes_map, ordered_nodes_map)
    sorted_graph = mace_pb2.NetDef()
    sorted_graph.tensors.extend(graph_def.tensors)
    sorted_graph.op.extend([node for node in ordered_nodes_map.values()])
    return sorted_graph
