import numpy as np
import matplotlib.pyplot as plt
from DBN import DBN
from RBM import RBM

def prin(*args, **kwargs):
    if 0:
        return print(*args, **kwargs)

def multiclass_log_loss(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -np.sum(y_true * np.log(y_pred), axis=1)
    mean_loss = np.mean(loss)
    return mean_loss
    
def softmax_calculation(rbm, entry_data):
        a = np.array(entry_data) @ rbm.W + rbm.b
        vec_exp = np.exp(a)
        results = vec_exp / vec_exp.sum(axis=1, keepdims=True)
        return results

class DNN:
    def __init__(self, size:int):
        assert len(size) >= 2
        self.DBN = DBN(size[:-1])
        self.classification_RBM = RBM(size[-2], size[-1])
        
    def pretrain_DNN(self, epochs:int, learning_rate, batch_size, entry_data):
        self.DBN.train(X=entry_data, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
        
    
    def in_out_network(self, entry_data):
        values_layers = (entry_data,)
        intermediate = entry_data
        
        for i, rbm in enumerate(self.DBN.RBNs):
            intermediate = np.array(rbm.in_out_RBM(intermediate))
            values_layers += (intermediate.copy(),)
        softmax = softmax_calculation(self.classification_RBM, intermediate)
        results = values_layers + (softmax.copy(),)
        assert len(results) == len(self.DBN.RBNs) + 2

        return results   
                    
    def retropropagation(self, epochs, learning_rate, batch_size, X, y, plot=True):
        y = np.array(y)
        assert len(X) == len(y), f"X et y must have the same sizes"
        plot_loss = []
        plot_error= []

        for epoch in range(epochs):
            permutation = np.random.permutation(y.shape[0])
            
            for batch in range(0, y.shape[0], batch_size):
                X_batch = X[permutation][batch:batch+batch_size]
                y_batch = y[permutation][batch:batch+batch_size]
                n = y_batch.shape[0]
                gradients = {}
                outputs = self.in_out_network(X_batch)
                L = len(outputs) - 1
                y_hat = outputs[-1]
                dz = y_hat - y_batch
                grad_b = dz.sum(axis=0) / n
                grad_W = (outputs[-2].T @ dz) / n
                W = [rbm.W for rbm in self.DBN.RBNs + [self.classification_RBM]]
                gradients[f"db{L}"], gradients[f"dw{L}"] = grad_b, grad_W

                for l in range(L-1, 0, -1):
                    x_l = outputs[l]
                    dz = (dz @ W[l].T) * x_l * (1-x_l)
                    grad_b = dz.sum(axis=0) / n
                    grad_W = (outputs[l-1].T @ dz) / n
                    gradients[f"db{l-1}"], gradients[f"dw{l-1}"] = grad_b, grad_W

                for l, rbm in enumerate(self.DBN.RBNs):
                    rbm.b -= learning_rate * gradients[f"db{l}"]
                    rbm.W -= learning_rate * gradients[f"dw{l}"]

                self.classification_RBM.b -= learning_rate * gradients[f"db{L}"]
                self.classification_RBM.W -= learning_rate * gradients[f"dw{L}"]

            values_layers = self.in_out_network(X)
            softmax = values_layers[-1]
            loss = multiclass_log_loss(y_true=y, y_pred=softmax)
            acc = self.test(X, y)
            plot_loss.append(loss)
            plot_error.append(acc)

        if plot:
            plt.plot(plot_loss, label="Multiclass Log loss")
            plt.legend()
            plt.show()
            plt.plot(plot_error, "r", label="Error Rate")
            plt.legend()
            plt.show()
    
    
    def test(self, X, y_true):
        assert len(X) == len(y_true), "X et y_true must have the same sizes"
        y_true = np.array(y_true)
        values_layers = self.in_out_network(X)
        softmax = values_layers[-1]
        y_hat = np.argmax(softmax, axis=1)
        y_true_argmax = np.argmax(y_true, axis=1)
        assert y_true_argmax.shape == y_hat.shape, f"{y_true_argmax.shape=} {y_hat.shape=}"
        assert len(y_true_argmax) == len(X)
        error_rate = (y_true_argmax!=y_hat).mean()
        
        return error_rate