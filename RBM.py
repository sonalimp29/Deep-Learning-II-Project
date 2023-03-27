import numpy as np
import matplotlib.pyplot as plt
from load_data import show


class RBM:
    def __init__(self, p, q, seed=0):
        self.rng = np.random.default_rng(seed)
        self.a = np.zeros(p)
        self.b = np.zeros(q)
        self.W = self.rng.normal(scale=0.1, size=(p, q))

    def in_out_RBM(self, X):
        return 1 / (1 + np.exp(-(X @ self.W + self.b)))

    def out_in_RBM(self, H):
        return 1 / (1 + np.exp(-(H @ self.W.T + self.a)))

    def train_RBM(self, X, learning_rate, epochs, batch_size):
        messages = 5
        samples = X.shape[0]
        rec_errors = []
        
        for i_epoch, epoch in enumerate(range(epochs)):
            self.rng.shuffle(X, axis=0)

            for batch in range(0, samples, batch_size):
                X_batch = X[batch:min(batch + batch_size, samples)]
                actual_batch_size = X_batch.shape[0]
                v_0 = X_batch
                p_h_v_0 = self.in_out_RBM(v_0)
                h_0 = (self.rng.uniform(size=p_h_v_0.shape) < p_h_v_0) * 1
                p_v_h_0 = self.out_in_RBM(h_0)
                v_1 = (self.rng.uniform(size=p_v_h_0.shape) < p_v_h_0) * 1
                p_h_v_1 = self.in_out_RBM(v_1)
                grad_a = np.sum(v_0 - v_1, axis=0)
                grad_b = np.sum(p_h_v_0 - p_h_v_1, axis=0)
                grad_w = v_0.T @ p_h_v_0 - v_1.T @ p_h_v_1
                self.W += learning_rate / actual_batch_size * grad_w
                self.a += learning_rate / actual_batch_size * grad_a
                self.b += learning_rate / actual_batch_size * grad_b

            H = self.in_out_RBM(X)
            X_rec = self.out_in_RBM(H)
            error = np.sum((X - X_rec) ** 2) / samples
            rec_errors.append(error)

            if messages:
                if i_epoch % round(epochs/messages) == 0:
                    print("Reconstruction error at epoch", epoch, "is", error)

        plt.plot(rec_errors)
        plt.title('Reconstruction error', fontsize=14)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Reconstruction error', fontsize=10)
        plt.show()

def generate_image_RBM(RBM, data:int, gibbs_iteartions:int, height:int, width:int, seed=0, start:bool=False):
    rng = np.random.default_rng(seed)
    p = len(RBM.a)
    q = len(RBM.b)
    generated_images = []

    for i in range(data):
        if start:
            h = (rng.uniform(size=q) < rng.uniform())

            for j in range(gibbs_iteartions - 1):
                v = (rng.uniform(size=p) < RBM.out_in_RBM(h)) * 1
                h = (rng.uniform(size=q) < RBM.in_out_RBM(v)) * 1

            v = (rng.uniform(size=p) < RBM.out_in_RBM(h)) * 1

        else:
            v = (rng.uniform(size=p) < rng.uniform())
            for j in range(gibbs_iteartions):

                h = (rng.uniform(size=q) < RBM.in_out_RBM(v)) * 1
                v = (rng.uniform(size=p) < RBM.out_in_RBM(h)) * 1

        v = v.reshape((height, width))
        generated_images.append(v)
        show(v)
        
    return generated_images
