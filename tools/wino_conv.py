import numpy as np
import math
import tensorflow as tf

A_T = np.array([[1, 1, 1, 0], [0, 1, -1, -1]]).astype(np.float32)
A = np.transpose(A_T)
B_T = np.array([
  [1, 0, -1, 0],
  [0, 1, 1, 0],
  [0, -1, 1, 0],
  [0, 1, 0, -1]
]).astype(np.float32)
B = np.transpose(B_T)
G = np.array([
  [1, 0, 0],
  [0.5, 0.5, 0.5],
  [0.5, -0.5, 0.5],
  [0, 0, 1],
]).astype(np.float32)
G_T = np.transpose(G)


def output_shape(input_shape, filter_shape):
  out_shape = np.zeros(4).astype(np.int32)
  out_shape[0] = input_shape[0]
  out_shape[1] = filter_shape[0]
  out_shape[2] = input_shape[2] - 2
  out_shape[3] = input_shape[3] - 2
  return out_shape


def winog_conv(input, filter):
  m = 2
  r = 3
  alpha = m + r - 1
  input_shape = input.shape
  filter_shape = filter.shape
  out_shape = output_shape(input_shape, filter_shape)

  K = filter_shape[0]
  C = input_shape[1]
  U = np.zeros((K * 16, C))

  for k in range(K):
    for c in range(C):
      u = np.dot(np.dot(G, filter[k, c, :, :]), G_T)
      for i in range(4):
        for j in range(4) :
          U[(i * 4 + j) * K + k, c] = u[i, j]

  print 'filter out: ', U.shape
  print U[0, 0]
  U.astype(np.float32).tofile("filter_out")

  rounded_h = int(math.ceil(out_shape[2] / 2.0))
  rounded_w = int(math.ceil(out_shape[3] / 2.0))
  P = input_shape[0] * rounded_h * rounded_w
  V = np.zeros((C * 16, P))
  for p in range(P):
    for c in range(C):
      n = p / (rounded_w * rounded_h)
      t = p % (rounded_h * rounded_w)
      h_idx = t / rounded_w
      w_idx = t % rounded_w
      h_start = h_idx * 2
      w_start = w_idx * 2
      h_end = min(h_start+4, input_shape[2])
      w_end = min(w_start+4, input_shape[3])
      d = np.zeros((4, 4))
      d[0:h_end-h_start, 0:w_end-w_start] = input[n, c, h_start:h_end, w_start:w_end]
      v = np.dot(np.dot(B_T, d), B)
      for i in range(4):
        for j in range(4):
          V[(i*4+j)*C + c, p] = v[i, j]

  tmp = V.reshape(16, C, P, 1)
  print 'input out: ', tmp.shape
  tmp.astype(np.float32).tofile("C")
  M = np.zeros((16 * K, P))
  for i in range(alpha * alpha):
    u = U[i * K : (i+1) * K, :]
    v = V[i * C : (i+1) * C, :]
    M[i * K : (i+1) * K, :] = np.dot(u, v)

  print 'M shape: ', M.shape
  M.astype(np.float32).tofile("gemm")
  res = np.zeros((out_shape[0], out_shape[2], out_shape[3], out_shape[1]))
  for k in range(K):
    for b in range(P):
      m = np.zeros((4, 4))
      for i in range(4):
        for j in range(4):
          m[i][j] = M[(i*4+j) * K + k, b]
      y = np.dot(np.dot(A_T, m), A)
      for i in range(2):
        for j in range(2):
          n = b / (rounded_h * rounded_w)
          t = b % (rounded_h * rounded_w)
          p = (t / rounded_w) * 2 + i
          q = (t % rounded_w) * 2 + j
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
  input = np.random.random([7, 61, 71, 31]).astype(np.float32)
  # input = np.fromfile(file="A", dtype=np.float32)
  # input = input.reshape(1, 3, 3, 5)
  print 'input shape: ', input.shape
  input.tofile("A")
  filter = np.random.random([3, 3, 31, 31]).astype(np.float32)
  tf_out = tf_conv(input, filter)
  input = input.transpose((0, 3, 1, 2))
  filter = filter.transpose((3, 2, 0, 1))
  print 'filter shape: ', filter.shape
  filter.tofile("filter_in")
  winog_out = winog_conv(input, filter)
  res = np.allclose(tf_out, winog_out)
  if res:
    print "=========Pass========="
  else:
    print "=========Failed========="
    print "TF: ", tf_out
    print "Winograd: ", winog_out


if __name__ == '__main__':
  main()

