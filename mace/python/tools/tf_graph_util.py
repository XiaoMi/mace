import tensorflow as tf
from collections import OrderedDict

def sort_graph_node(node, nodes_map, ordered_nodes_map):
    if node.name not in ordered_nodes_map:
        for input_tensor_name in node.input:
            input_node_name = input_tensor_name.split(':')[
                0] if ':' in input_tensor_name else input_tensor_name
            if input_node_name not in nodes_map or input_node_name in ordered_nodes_map:
                continue

            input_node = nodes_map[input_node_name]
            sort_graph_node(input_node, nodes_map, ordered_nodes_map)
            ordered_nodes_map[input_node_name] = input_node
        ordered_nodes_map[node.name] = node

def sort_graph(graph_def):
    nodes_map = {}
    ordered_nodes_map = OrderedDict()
    for node in graph_def.node:
        nodes_map[node.name] = node
    for node in graph_def.node:
        sort_graph_node(node, nodes_map, ordered_nodes_map)
    sorted_graph = tf.GraphDef()
    sorted_graph.node.extend([node for _, node in ordered_nodes_map.iteritems()])
    return sorted_graph