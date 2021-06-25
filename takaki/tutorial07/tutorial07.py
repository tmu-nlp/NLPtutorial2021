from collections import defaultdict
import numpy as np


class NlpTutorial7:

    def forward(self, network, phi0):
        phi = [ phi0 ]
        for i in range(len(network)):
            w, b = network[i]
            phi[i] = np.tanh(np.dot(w, phi[i-1]) + b)
        return phi

    def backward(self, net, phi, yn):
        j  = len(net)
        d  = np.zeros(j+1, dtype=np.ndarray)
        dn = np.array([float(yn) - phi[j][0]])
        for i in reversed(reverse(j)):
            dn[i+1] = d[i+1] * (1 - phi[i+1] ** 2)
            w, b = net[i]
            d[i] = np.dot(dn[i+1], w)
        return dn

    def update_weights(self, net, phi, dn, l):
        for i in range(len(net)):
            w, b = net[i]
            w += l * np.outer(dn[i+1], phi[i])
            b += l * dn[i+1]

    def create_features(self, line):
        phi = defaultdict(int)
        for word in line.split():
            phi[f"UNI:{word}"] += 1
        return phi

    def train(self, iters, l):
        ids = defaultdict(lambda: len(ids))
        feat_lab = []
        for x, y in data.items():
            feat_lab.append(this.create_features(x), y)

        for i in iters:
            for phi0, y in feat_lab:
                phi = self.forward(net, phi0)
                dn  = self.backward(net, phi, y)
                self.update_weights(net, phi, dn, l)
