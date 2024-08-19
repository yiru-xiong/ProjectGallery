#########################################
####### A Mini Torch from Scratch #######
#########################################
# framework for Neural Networks
import numpy as np


class Module:
    def __init__(self):
        super().__init__()
        self.params = dict()
        self.grads = dict()
        self.children = dict()
        self.cache = dict()

    def _register_param(self, name: str, param: np.ndarray):
        assert isinstance(param, np.ndarray)
        self.params[name] = param
        self.grads[name] = np.zeros_like(param)

    def _register_child(self, name: str, child: 'Module'):
        assert isinstance(child, Module)
        self.children[name] = child

    def forward(self, *x):
        raise NotImplementedError

    def backward(self, *g):
        raise NotImplementedError

    def named_parameters(self, base: tuple = ()):
        assert self.params.keys() == self.grads.keys()
        for name in self.params:
            full_name = '.'.join(base + (name,))
            yield (full_name, self.params[name], self.grads[name])

        for child_name, child in self.children.items():
            yield from child.named_parameters(base=base + (child_name,))

    def clear_cache(self):

        self.cache.clear()
        for child in self.children.values():
            child.clear_cache()


def sigmoid(x):
    """
    :param x: np.ndarray
    :return: np.ndarray, same shape as x, elementwise sigmoid of x
    """
    return 1/(1+np.exp(-x))


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: np.ndarray, shape (N, in_features)
        :return: np.ndarry, elementwise sigmoid of input, same shape as x.

        this layer computes \vec{z} given \vec{a}.
        """
        self.sig = sigmoid(x)

        self.cache['sig'] = self.sig
        
        return self.sig

    def backward(self, g):
        """
        :param g: np.ndarray, shape (N, in_features), the gradient of
               loss w.r.t. output of this layer.
        :return: np.ndarray, shape (N, in_features), the gradient of
                 loss w.r.t. input of this layer.
        """
        z = self.cache['sig']
        da = np.multiply(np.multiply(z,(1-z)),g)
        self.cache['da'] = da
        return da
        


class Linear(Module):
    def __init__(self, weight, bias):
        super().__init__()
        # weight has shape (out_features, in_features)
        self._register_param('weight', weight)
        # bias has shape (out_features,)
        self._register_param('bias', bias)

    def forward(self, x):
        """input has shape (N, in_features)
        and output has shape (N, out_features)

        this layer computes \vec{a} given \vec{x},
        or \vec{b} given \vec{z}.
        """
        w = self.params['weight']
        bias = self.params['bias']
        lc = np.dot(x, w.T) + bias
        self.cache['l_input']= x
        self.cache['l_output']= lc
        return lc
        

    def backward(self, g):
        """g is of shape (N, out_features)
        g_input should be of shape (N, in_features)"""
        w = self.params['weight']
        l_input = self.cache['l_input']
        dweight = np.dot(g.T,l_input)
        dbias = g.sum(axis=0)
        #dinput = gw * g.reshape(-1,1)
        dinput = np.dot(g,w)
        self.cache['dweight'] = dweight
        self.cache['dbias'] = dbias
        self.grads['weight'] += dweight
        self.grads['bias'] += dbias
        return dinput
        
        
class CrossEntropyLoss(Module):
    """softmax + cross entropy loss"""

    def __init__(self):
        super().__init__()

    def forward(self, score, label):
        """
        :param score: np.ndarray, shape (N, C), the score values for
               input of softmax ($b_k$ in the writeup).
        :param label: integer-valued np.ndarray, shape (N,), all in [0,C-1]
               (non-zero idx of $\vec{y}$ in the writeup).
        :return: the mean negative cross entropy loss
               ($J(\alpha, \beta)$ in the write up).
        """
        N = score.shape[0]
        C = score.shape[1]
        label_pos = np.array(label).reshape(-1)
        #one-hot encoding 
        y = np.eye(C)[label_pos]
        sum_score = np.sum(np.exp(score),axis=1)
        y_hat = np.array([(np.exp(score[i])/sum_score[i]) for i in range(N)])
        Loss = np.sum(np.multiply(y,np.log(y_hat)))*(-1/N)
        self.cache['N'] = N
        self.cache['y']=y
        self.cache['y_hat']= y_hat
        self.cache['loss']= Loss
        return Loss 

    def backward(self):
        """returns the gradient of loss w.r.t. `score`"""
        #dl/d_yhat 
        N = self.cache['N']
        y_hat = self.cache['y_hat']
        y = self.cache['y']
        #(dl/d_yhat) * (d_yhat/db)
        db = (y_hat - y)/N
        return db