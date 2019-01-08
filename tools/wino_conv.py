# Copyright 2018 The MACE Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import math
import tensorflow as tf

A_T = {}
A = {}
B_T = {}
B = {}
G = {}
G_T = {}
# f(2, 3)
A_T[4] = np.array([[1, 1, 1, 0], [0, 1, -1, -1]]).astype(np.float32)
A[4] = np.transpose(A_T[4])
B_T[4] = np.array([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0],
                   [0, 1, 0, -1]]).astype(np.float32)
B[4] = np.transpose(B_T[4])
G[4] = np.array([
    [1, 0, 0],
    [0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0, 0, 1],
]).astype(np.float32)
G_T[4] = np.transpose(G[4])

# f(4, 3)
A_T[6] = np.array([
    [1, 1, 1, 1, 1, 0],
    [0, 1, -1, 2, -2, 0],
    [0, 1, 1, 4, 4, 0],
    [0, 1, -1, 8, -8, 1],
]).astype(np.float32)
A[6] = np.transpose(A_T[6])
B_T[6] = np.array([
    [4, 0, -5, 0, 1, 0],
    [0, -4, -4, 1, 1, 0],
    [0, 4, -4, -1, 1, 0],
    [0, -2, -1, 2, 1, 0],
    [0, 2, -1, -2, 1, 0],
    [0, 4, 0, -5, 0, 1],
]).astype(np.float32)
B[6] = np.transpose(B_T[6])
G[6] = np.array([
    [1 / 4.0, 0, 0],
    [-1 / 6.0, -1 / 6.0, -1 / 6.0],
    [-1 / 6.0, 1 / 6.0, -1 / 6.0],
    [1 / 24.0, 1 / 12.0, 1 / 6.0],
    [1 / 24.0, -1 / 12.0, 1 / 6.0],
    [0, 0, 1],
]).astype(np.float32)
G_T[6] = np.transpose(G[6])

# f(6, 3)
A_T[8] = np.array([
    [1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, -1, 2, -2, 1 / 2., -1 / 2., 0],
    [0, 1, 1, 4, 4, 1 / 4., 1 / 4., 0],
    [0, 1, -1, 8, -8, 1 / 8., -1 / 8., 0],
    [0, 1, 1, 16, 16, 1 / 16., 1 / 16., 0],
    [0, 1, -1, 32, -32, 1 / 32., -1 / 32., 1],
]).astype(np.float32)
A[8] = np.transpose(A_T[8])
B_T[8] = np.array([
    [1, 0, -21 / 4., 0, 21 / 4., 0, -1, 0],
    [0, 1, 1, -17 / 4., -17 / 4., 1, 1, 0],
    [0, -1, 1, 17 / 4., -17 / 4., -1, 1, 0],
    [0, 1 / 2., 1 / 4., -5 / 2., -5 / 4., 2, 1, 0],
    [0, -1 / 2., 1 / 4., 5 / 2., -5 / 4., -2, 1, 0],
    [0, 2, 4, -5 / 2., -5, 1 / 2., 1, 0],
    [0, -2, 4, 5 / 2., -5, -1 / 2., 1, 0],
    [0, -1, 0, 21 / 4., 0, -21 / 4., 0, 1],
]).astype(np.float32)
B[8] = np.transpose(B_T[8])
G[8] = np.array([
    [1, 0, 0],
    [-2 / 9., -2 / 9., -2 / 9.],
    [-2 / 9., 2 / 9., -2 / 9.],
    [1 / 90., 1 / 45., 2 / 45.],
    [1 / 90., -1 / 45., 2 / 45.],
    [32 / 45., 16 / 45., 8 / 45.],
    [32 / 45., -16 / 45., 8 / 45.],
    [0, 0, 1],
]).astype(np.float32)
G_T[8] = np.transpose(G[8])


def output_shape(input_shape, filter_shape):
    out_shape = np.zeros(4).astype(np.int32)
    out_shape[0] = input_shape[0]
    out_shape[1] = filter_shape[0]
    out_shape[2] = input_shape[2] - 2
    out_shape[3] = input_shape[3] - 2
    return out_shape


def winograd_conv(m, r, input, filter):
    alpha = m + r - 1
    print 'Winograd(m = %d, r = %d, tile size=%d' % (m, r, alpha)
    alpha_square = alpha * alpha
    input_shape = input.shape
    filter_shape = filter.shape
    out_shape = output_shape(input_shape, filter_shape)

    K = filter_shape[0]
    C = input_shape[1]
    U = np.zeros((K * alpha_square, C))

    for k in range(K):
        for c in range(C):
            u = np.dot(np.dot(G[alpha], filter[k, c, :, :]), G_T[alpha])
            for i in range(alpha):
                for j in range(alpha):
                    U[(i * alpha + j) * K + k, c] = u[i, j]

    print 'filter out: ', U.shape

    rounded_h = int(math.ceil(out_shape[2] / (m * 1.0)))
    rounded_w = int(math.ceil(out_shape[3] / (m * 1.0)))
    P = input_shape[0] * rounded_h * rounded_w
    V = np.zeros((C * alpha_square, P))
    for p in range(P):
        for c in range(C):
            n = p / (rounded_w * rounded_h)
            t = p % (rounded_h * rounded_w)
            h_idx = t / rounded_w
            w_idx = t % rounded_w
            h_start = h_idx * m
            w_start = w_idx * m
            h_end = min(h_start + alpha, input_shape[2])
            w_end = min(w_start + alpha, input_shape[3])
            d = np.zeros((alpha, alpha))
            d[0:h_end-h_start, 0:w_end-w_start] = \
                input[n, c, h_start:h_end, w_start:w_end]
            v = np.dot(np.dot(B_T[alpha], d), B[alpha])
            for i in range(alpha):
                for j in range(alpha):
                    V[(i * alpha + j) * C + c, p] = v[i, j]

    tmp = V.reshape(alpha_square, C, P, 1)
    print 'input out: ', tmp.shape
    tmp.astype(np.float32).tofile("C")
    M = np.zeros((alpha_square * K, P))
    for i in range(alpha_square):
        u = U[i * K:(i + 1) * K, :]
        v = V[i * C:(i + 1) * C, :]
        M[i * K:(i + 1) * K, :] = np.dot(u, v)

    print 'M shape: ', M.shape
    M.astype(np.float32).tofile("gemm")
    res = np.zeros((out_shape[0], out_shape[2], out_shape[3], out_shape[1]))
    for k in range(K):
        for b in range(P):
            tm = np.zeros((alpha, alpha))
            for i in range(alpha):
                for j in range(alpha):
                    tm[i][j] = M[(i * alpha + j) * K + k, b]
            y = np.dot(np.dot(A_T[alpha], tm), A[alpha])
            for i in range(m):
                for j in range(m):
                    n = b / (rounded_h * rounded_w)
                    t = b % (rounded_h * rounded_w)
                    p = (t / rounded_w) * m + i
                    q = (t % rounded_w) * m + j
                    if p >= out_shape[2] or q >= out_shape[3]:
                        continue
                    res[n, p, q, k] = y[i, j]

    print 'Res shape: ', res.shape
    res.astype(np.float32).tofile("res")

    return res


def tf_conv(input, filter):
    conv_op = tf.nn.conv2d(input, filter, [1, 1, 1, 1], 'VALID')
    with tf.Session() as sess:
        res = sess.run(conv_op)
    return res


def main():
    input = np.random.random([5, 23, 29, 15]).astype(np.float32)
    # input = np.fromfile(file="A", dtype=np.float32)
    # input = input.reshape(1, 3, 3, 5)
    print 'input shape: ', input.shape
    # input.tofile("A")
    filter = np.random.random([3, 3, 15, 13]).astype(np.float32)
    tf_out = tf_conv(input, filter)
    input = input.transpose((0, 3, 1, 2))
    filter = filter.transpose((3, 2, 0, 1))
    print 'filter shape: ', filter.shape
    # filter.tofile("filter_in")
    for i in [2, 4, 6]:
        print "==========f(%d,3)==========" % i
        winograd_out = winograd_conv(i, 3, input, filter)
        res = np.allclose(tf_out, winograd_out)
        if res:
            print "=========Pass========="
        else:
            print "=========Failed======="
            print "TF: ", tf_out
            print "Winograd: ", winograd_out


if __name__ == '__main__':
    main()
