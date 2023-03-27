import numpy as np
import matplotlib.pyplot as plt
from RBM import RBM, generate_image_RBM
from load_data import show


class DBN:
    def __init__(self, size):
        self.RBNs = []
        
        for i in range(len(size)-1):
            self.RBNs.append(RBM(size[i], size[i+1]))
    
    def train(self, X, epochs:int, learning_rate, batch_size:int):
        entrance = X

        for i, rbn in enumerate(self.RBNs):
            rbn.train_RBM(entrance, learning_rate, epochs=epochs, batch_size=batch_size)
            entrance = rbn.in_out_RBM(entrance)
        
    def generate_image_DBN(self, gibbs_iteartions:int, data:int, height:int, width:int, seed=0):
        assert len(self.RBNs[0].a) == height * width, f"Cannot create image of size {height}x{width}={height * width} px"
        start=False
        current=[]
        rng = np.random.default_rng(seed)
        last_rbm = self.RBNs[-1]
        p = len(last_rbm.a)
        q = len(last_rbm.b)

        for i in range(data):
            if start:
                h = (rng.uniform(size=q) < rng.uniform())

                for j in range(gibbs_iteartions - 1):
                    v = (rng.uniform(size=p) < last_rbm.out_in_RBM(h)) * 1
                    h = (rng.uniform(size=q) < last_rbm.in_out_RBM(v)) * 1

                v = (rng.uniform(size=p) < last_rbm.out_in_RBM(h)) * 1

            else:
                v = (rng.uniform(size=p) < rng.uniform())

                for j in range(gibbs_iteartions):
                    h = (rng.uniform(size=q) < last_rbm.in_out_RBM(v)) * 1
                    v = (rng.uniform(size=p) < last_rbm.out_in_RBM(h)) * 1

            current.append(v)
        
        for rbn in reversed(self.RBNs[:-1]):
            current = [rbn.out_in_RBM(x) for x in current]
        
        images = [v.reshape((height, width)) for v in current]
        
        for image in images:
            show(image)

        return images
            
            