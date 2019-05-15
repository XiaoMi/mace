import json
import numpy as np

from google.protobuf.json_format import _Printer

THREASHOLD = 16


class NPEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class ModelVisualizer(object):
    def __init__(self, model_name, proto):
        self._output_file = "build/%s_index.html" % model_name
        self._proto = proto

    def render_html(self):
        json_obj = {
            "nodes": [],
            "links": []
        }

        json_printer = _Printer()

        for op in self._proto.op:
            op_json = json_printer._MessageToJsonObject(op)
            op_json["id"] = op_json["name"]
            op_json["node_type"] = "op"
            json_obj["nodes"].append(op_json)

        for tensor in self._proto.tensors:
            tensor_json = json_printer._MessageToJsonObject(tensor)

            tensor_json["id"] = tensor_json["name"]
            if "floatData" in tensor_json and \
                    len(tensor_json["floatData"]) > THREASHOLD:
                del tensor_json["floatData"]
            if "int32Data" in tensor_json and \
                    len(tensor_json["int32Data"]) > THREASHOLD:
                del tensor_json["int32Data"]
            tensor_json["node_type"] = "tensor"
            json_obj["nodes"].append(tensor_json)

        node_ids = [node["id"] for node in json_obj["nodes"]]

        tensor_to_op = {}
        for op in self._proto.op:
            for tensor in op.output:
                tensor_to_op[tensor] = op.name

        for op in json_obj["nodes"]:
            if "input" in op:
                for input in op["input"]:
                    if input in node_ids and op["name"] in node_ids:
                        # for weights
                        json_obj["links"].append(
                            {"source": input, "target": op["name"]})
                    elif input in tensor_to_op and \
                            tensor_to_op[input] in node_ids:
                        # for intermediate tensor
                        json_obj["links"].append(
                            {"source": tensor_to_op[input],
                             "target": op["name"]})
                    else:
                        # for input
                        json_obj["nodes"].append({
                            "id": input,
                            "name": input,
                            "node_type": "input"
                        })
                        json_obj["links"].append(
                            {"source": input, "target": op["name"]})

        json_msg = json.dumps(json_obj, cls=NPEncoder)

        with open("mace/python/tools/visualization/index.html") as f:
            html = f.read()
            return html % json_msg

    def save_html(self):
        html = self.render_html()
        with open(self._output_file, "wb") as f:
            f.write(html)
