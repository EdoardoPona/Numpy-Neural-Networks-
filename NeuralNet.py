import numpy as np
import random
import matplotlib.pyplot as plt


def one_hot(i):
    out = np.zeros((10))
    out[i] = 1
    return out


def shuffle_list(*ls):
    l =list(zip(*ls))
    random.shuffle(l)
    return zip(*l)


data = open('train.csv', 'r').readlines()
targets = []
del data[0]

print('loading dataset')
for i, line in enumerate(data):
    data[i] = np.array([int(j) for j in line.split(',')][1:])
    targets.append(one_hot(int(line[0])))
print('done')


def get_batch(b_size):
    i = random.randint(0, len(data)-b_size)
    return np.array(data[i:i+b_size]), np.array(targets[i:i+b_size])


def get_accuracy(output, targets, batch_size):
    correct = 0
    for o, t in zip(output, targets):
        a = np.argmax(o, 0)
        b = np.argmax(t, 0)

        if a == b:
            correct += 1

    return correct / batch_size


def init_weight(shape):
    return np.random.randn(shape[0], shape[1]) * np.sqrt(1 / shape[0])


def init_bias(size):
    return np.zeros(size)


def sigmoid(x):
    return 1 / (1 + np.exp(-x.astype(np.float128)))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(X):             # stable version
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


def cross_entropy(predictions, targets, epsilon=1e-12):

    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(np.sum(targets*np.log(predictions+1e-9)))/N
    return ce


def mse(y, t):
    return np.sum(1/2*(y - t)**2)


class BatchNormalization:
    def __init__(self, gamma):
        # no beta as there is a bias in the network already
        self.gamma = gamma
        self.eps = 1e-8

    def forwards(self, x):
        self.n, self.d = x.shape
        mean = 1/self.n * np.sum(x, axis=0)

        self.xm = x - mean
        x_sq = self.xm**2
        self.variance = 1/self.n * np.sum(x_sq, axis=0)

        self.std = np.sqrt(self.variance + 1e-8)
        self.i_std = 1/self.std

        self.x_ = self.xm * self.i_std

        out = self.gamma*self.x_
        return out

    def backwards(self):

        d_gamma = self.x_

        dx_ = self.gamma
        distd = dx_ * self.xm
        dstd = distd * -1/self.variance
        dvariance = 0.5*1/self.std*dstd
        dsq = 1/self.n * np.ones((self.n, self.d))

        dxm1 = dx_ * distd
        dxm2 = 2*self.xm*dsq

        dx1 = dxm1 + dxm2
        dm = -np.sum((dxm1 + dxm2), axis=0)
        dx2 = 1/self.n * np.ones((self.n, self.d))

        dx = dx1 + dx2

        return dx, d_gamma


class NeuralNetwork:

    def __init__(self):
        self.w0 = init_weight((784, 512))
        self.b0 = init_bias(512)

        self.w1 = init_weight((512, 128))
        self.b1 = init_bias(128)

        self.batch_norm = BatchNormalization(1)

        self.w2 = init_weight((128, 10))
        self.b2 = init_bias(10)

        self.B2 = init_weight((10, 128))           # replaces self.w2.T if using feedback training
        self.B1 = init_weight((128, 512))           # replaces self.w1.T

    def forward(self, x):
        self.x = x

        self.a0 = np.dot(x, self.w0)
        self.h0 = sigmoid(self.a0)

        self.h0_norm = self.batch_norm.forwards(self.h0)

        self.a1 = np.dot(self.h0_norm, self.w1)
        self.h1 = sigmoid(self.a1)

        self.a2 = np.dot(self.h1, self.w2)
        self.out = self.a2

        return self.out

    def backwards(self, t):
        e = (self.out - t)    # * d_sigmoid(self.a2)    # when using mse

        dJdw2 = np.dot(self.h1.T, e)
        dJdb2 = np.sum(e, axis=0)

        delta1 = np.dot(e * d_sigmoid(self.a2), self.w2.T) * d_sigmoid(self.a1)
        dJdw1 = np.dot(self.h0_norm.T, delta1)
        dJdb1 = np.sum(delta1, axis=0)

        d_h0norm_dh0, d_gamma = self.batch_norm.backwards()

        delta0 = np.dot(delta1, self.w1.T) * d_h0norm_dh0 * d_sigmoid(self.a0)
        dJdw0 = np.dot(self.x.T, delta0)
        dJdb0 = np.sum(delta0, axis=0)

        return ((dJdw0, dJdb0), (dJdw1, dJdb1), (dJdw2, dJdb2)), d_gamma


class ShallowNeuralNetwork:

    def __init__(self):
        self.w0 = init_weight((784, 256))
        self.b0 = init_bias(256)

        self.w1 = init_weight((256, 10))
        self.b1 = init_bias(10)

        self.B1 = init_weight((256, 512))

    def forward(self, x):
        self.x = x

        self.a0 = np.dot(x, self.w0)
        self.h0 = sigmoid(self.a0)

        self.a1 = np.dot(self.h0, self.w1)
        self.out = self.a1

        return self.out

    def backwards(self, t):
        e = (self.out - t) * d_sigmoid(self.a1)    # when using mse

        dJdw1 = np.dot(self.h0.T, e)
        dJdb1 = np.sum(e, axis=0)

        # delta0 = np.dot(e * d_sigmoid(self.a1), self.B1) * d_sigmoid(self.a0)
        delta0 = np.dot(e * d_sigmoid(self.a1), self.w1.T) * d_sigmoid(self.a0)
        dJdw0 = np.dot(self.x.T, delta0)
        dJdb0 = np.sum(delta0, axis=0)

        return (dJdw0, dJdb0), (dJdw1, dJdb1)


network = ShallowNeuralNetwork()
n_layers = 2

costs = []

batch_size = 50
lr = 1e-3


def train_SGD(iter_num):
    for i in range(int(iter_num)):
        b, t = get_batch(batch_size)

        out = network.forward(b)
        cost = cross_entropy(out, t)

        # gradients, d_gamma = network.backwards(t)
        gradients = network.backwards(t)

        network.w0 -= lr * gradients[0][0]
        network.w1 -= lr * gradients[1][0]
        # network.w2 -= lr * gradients[2][0]

        network.b0 -= lr * gradients[0][1]
        network.b1 -= lr * gradients[1][1]
        # network.b2 -= lr * gradients[2][1]

        # network.batch_norm.gamma -= lr * d_gamma

        if i % 10 == 0:
            print(i, cost, get_accuracy(out, t, batch_size))
            costs.append(cost)


def train_SGD_momentum(iter_num, momentum=0.9):
    updates = [(0, 0) for i in range(n_layers)]     # [w_update, b_update] for every layer
    # gamma_update = 0            # for the gamma parameter in batch normalization
    for i in range(int(iter_num)):
        b, t = get_batch(batch_size)

        out = network.forward(b)
        cost = cross_entropy(out, t)

        # gradients, d_gamma = network.backwards(t)
        gradients = network.backwards(t)

        # gamma_update = lr*d_gamma + momentum*gamma_update
        updates = [(lr*gradients[i][0] + momentum*updates[i][0], lr*gradients[i][1] + momentum*updates[i][1]) for i in range(n_layers)]

        network.w0 -= updates[0][0]
        network.w1 -= updates[1][0]
        # network.w2 -= updates[2][0]

        network.b0 -= updates[0][1]
        network.b1 -= updates[1][1]
        # network.b2 -= updates[2][1]

        # network.batch_norm.gamma -= gamma_update

        if i % 10 == 0:
            print(i, cost, get_accuracy(out, t, batch_size))
            costs.append(cost)


def train_SGD_nesterov(iter_num, momentum=0.9):
    updates = [(0, 0) for i in range(n_layers)]     # [w_update, b_update] for every layer
    # gamma_update = 0        # gamma parameter in batch norm
    for i in range(int(iter_num)):
        b, t = get_batch(batch_size)

        # approximating future parameters
        network.w0 -= momentum * updates[0][0]
        network.w1 -= momentum * updates[1][0]
        # network.w2 -= momentum * updates[2][0]

        network.b0 -= momentum * updates[0][1]
        network.b1 -= momentum * updates[1][1]
        # network.b2 -= momentum * updates[2][1]

        # network.batch_norm.gamma -= momentum * gamma_update

        out = network.forward(b)
        cost = cross_entropy(out, t)

        # gradients, d_gamma = network.backwards(t)
        gradients = network.backwards(t)

        network.w0 -= lr * gradients[0][0]
        network.w1 -= lr * gradients[1][0]
        # network.w2 -= lr * gradients[2][0]

        network.b0 -= lr * gradients[0][1]
        network.b1 -= lr * gradients[1][1]
        # network.b2 -= lr * gradients[2][1]

        # network.batch_norm.gamma -= lr * d_gamma

        updates = [(lr*gradients[i][0] + momentum*updates[i][0], lr*gradients[i][1] + momentum*updates[i][1]) for i in range(n_layers)]
        # gamma_update = lr*d_gamma + momentum*gamma_update

        if i % 10 == 0:
            print(i, cost, get_accuracy(out, t, batch_size))
            costs.append(cost)


def train_AdaGrad(iter_num):
    """ only works without batch normalization for now (shallow network) """
    r = [[0, 0] for i in range(n_layers)]

    lr_matrix_w0 = lr * np.ones(network.w0.shape)
    lr_matrix_b0 = lr * np.ones(network.b0.shape)

    lr_matrix_w1 = lr * np.ones(network.w1.shape)
    lr_matrix_b1 = lr * np.ones(network.b1.shape)

    for i in range(int(iter_num)):
        b, t = get_batch(batch_size)

        out = network.forward(b)
        cost = cross_entropy(out, t)

        gradients = network.backwards(t)

        for j in range(n_layers):
            r[j][0] += np.multiply(gradients[j][0], gradients[j][0])
            r[j][1] += np.multiply(gradients[j][1], gradients[j][1])

        network.w0 -= np.multiply(np.divide(lr_matrix_w0, 1e-7 + np.sqrt(r[0][0])), gradients[0][0])
        network.b0 -= np.multiply(np.divide(lr_matrix_b0, 1e-7 + np.sqrt(r[0][1])), gradients[0][1])

        network.w1 -= np.multiply(np.divide(lr_matrix_w1, 1e-7 + np.sqrt(r[1][0])), gradients[1][0])
        network.b1 -= np.multiply(np.divide(lr_matrix_b1, 1e-7 + np.sqrt(r[1][1])), gradients[1][1])

        if i % 10 == 0:
            print(i, cost, get_accuracy(out, t, batch_size))
            costs.append(cost)


def train_RMSProp(iter_num, decay_rate=0.5):
    """ only works without batch normalization for now (shallow network) """
    r = [[0, 0] for i in range(n_layers)]

    for i in range(int(iter_num)):
        b, t = get_batch(batch_size)

        out = network.forward(b)
        cost = cross_entropy(out, t)

        gradients = network.backwards(t)

        for j in range(n_layers):
            r[j][0] = decay_rate * r[j][0] + np.multiply((1 - decay_rate) * gradients[j][0], gradients[j][0])
            r[j][1] = decay_rate * r[j][1] + np.multiply((1 - decay_rate) * gradients[j][1], gradients[j][1])

        network.w0 -= np.multiply(np.divide(lr, 1e-7 + np.sqrt(r[0][0])), gradients[0][0])
        network.b0 -= np.multiply(np.divide(lr, 1e-7 + np.sqrt(r[0][1])), gradients[0][1])
        network.w1 -= np.multiply(np.divide(lr, 1e-7 + np.sqrt(r[1][0])), gradients[1][0])
        network.b1 -= np.multiply(np.divide(lr, 1e-7 + np.sqrt(r[1][1])), gradients[1][1])

        if i % 10 == 0:
            print(i, cost, get_accuracy(out, t, batch_size))
            costs.append(cost)


def train_RMSProp_nesterov(iter_num, momentum=0.9, decay_rate=0.5):
    """ only works without batch normalization for now (shallow network) """
    updates = [[0, 0] for i in range(n_layers)]
    r = [[0, 0] for i in range(n_layers)]

    for i in range(int(iter_num)):
        b, t = get_batch(batch_size)

        network.w0 += momentum * updates[0][0]
        network.b0 += momentum * updates[0][1]
        network.w1 += momentum * updates[1][0]
        network.b1 += momentum * updates[1][1]

        out = network.forward(b)
        cost = cross_entropy(out, t)

        gradients = network.backwards(t)

        for j in range(n_layers):
            r[j][0] = decay_rate * r[j][0] + np.multiply((1 - decay_rate) * gradients[j][0], gradients[j][0])
            r[j][1] = decay_rate * r[j][1] + np.multiply((1 - decay_rate) * gradients[j][1], gradients[j][1])

        network.w0 -= np.multiply(np.divide(lr, 1e-7+np.sqrt(r[0][0])), gradients[0][0])
        network.b0 -= np.multiply(np.divide(lr, 1e-7+np.sqrt(r[0][1])), gradients[0][1])
        network.w1 -= np.multiply(np.divide(lr, 1e-7+np.sqrt(r[1][0])), gradients[1][0])
        network.b1 -= np.multiply(np.divide(lr, 1e-7+np.sqrt(r[1][1])), gradients[1][1])

        # updating the updates
        updates[0][0] = momentum * updates[0][0] - np.multiply(np.divide(lr, 1e-7+np.sqrt(r[0][0])), gradients[0][0])
        updates[0][1] = momentum * updates[0][1] - np.multiply(np.divide(lr, 1e-7+np.sqrt(r[0][1])), gradients[0][1])
        updates[1][0] = momentum * updates[1][0] - np.multiply(np.divide(lr, 1e-7+np.sqrt(r[1][0])), gradients[1][0])
        updates[1][1] = momentum * updates[1][1] - np.multiply(np.divide(lr, 1e-7+np.sqrt(r[1][1])), gradients[1][1])

        if i % 10 == 0:
            print(i, cost, get_accuracy(out, t, batch_size))
            costs.append(cost)


train_RMSProp_nesterov0(300)
plt.plot(list(range(len(costs))), costs)
plt.show()
