import json
import socket
import itertools


class FalconCli(object):
    def __init__(self, addr, debug=True, buf_size=1000):
        self.socket_ = socket.create_connection(addr)
        self.stream = self.socket_.makefile()
        self.id_counter = itertools.count()
        self.debug = debug
        self.buf_size = buf_size

    def __del__(self):
        self.socket_.close()
        self.stream.close()

    @classmethod
    def connect(cls,
                server="transfer.falcon.miliao.srv",
                port=8433,
                debug=True,
                buf_size=1000):
        try:
            return FalconCli((server, port), debug, buf_size)
        except socket.error, exc:
            print "error: connect to %s:%s error: %s" % (server, port, exc)

    def call(self, name, *params):
        request = dict(
            id=next(self.id_counter), params=list(params), method=name)
        payload = json.dumps(request).encode()
        if self.debug:
            print "--> req:", payload
        self.socket_.sendall(payload)

        response = self.stream.readline()
        if not response:
            raise Exception('empty response')

        if self.debug:
            print "<-- resp:", response

        response = json.loads(response.decode("utf8"))
        if response.get('error') is not None:
            raise Exception(response.get('error'))

        return response.get('result')

    def update(self, lines):
        s = 0
        resp = []

        while True:
            buf = lines[s:s + self.buf_size]
            s = s + self.buf_size
            if len(buf) == 0:
                break
            r = self.call("Transfer.Update", buf)
            resp.append(r)

        return resp
