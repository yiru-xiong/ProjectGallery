#########################################
####### A Mini Torch from Scratch #######
#########################################

import numpy as np

from minitorch_lib import Linear, Sigmoid, CrossEntropyLoss, Module

def compute_linear(x: np.ndarray,
                   weight: np.ndarray,
                   bias: np.ndarray,
                   grad_higher: np.ndarray):
    """
    :param x: np.ndarray, shape (N, in_features), input of the linear layer
    :param weight: np.ndarray, shape (output_feature, in_feature), the
                    weight matrix of the linear layer
    :param bias: np.ndarray, shape (output_feature,), the bias term added
                to the output
    :param grad_higher: np.ndarray, shape (N, out_features), the gradient
                of loss w.r.t. output of this layer

    :return: should be a dictionary containing following keys:

         'output':  np.ndarray, shape (N, out_features), the output of the
                    linear layer

         'grad_input':  np.ndarray, shape (N, in_features), the gradient
                        of loss w.r.t. input of this layer

         'grad_weight':  np.ndarray, shape (out_features, in_features),
                         the gradient of loss w.r.t. weight matrix of
                         this layer

         'grad_bias':  np.ndarray, shape (out_features,), the gradient
                       of loss w.r.t. bias of this layer
    """
    linear_layer=Linear(weight, bias)
    l_output = linear_layer.forward(x)

    l_grad_input = linear_layer.backward(grad_higher)

    l_grad_weight = linear_layer.cache['dweight']    

    l_grad_bias = linear_layer.cache['dbias']
    
    linear_output = {'output':l_output, 'grad_input':l_grad_input,'grad_weight':l_grad_weight,'grad_bias':l_grad_bias}  
    return linear_output


def compute_sigmoid(x: np.ndarray, grad_higher: np.ndarray):
    """
    :param x: np.ndarray, shape (N, in_features), input of the sigmoid layer
    :param grad_higher: np.ndarray, shape (N, in_features), the gradient
                        of loss w.r.t. the output of this layer

    :return: should be a dictionary containing following keys:

        'output':  np.ndarray, shape (N, in_features), output of the
                   sigmoid layer

        'grad_input':  np.ndarray, shape (N, in_features), the gradient
                       of loss w.r.t. the input of this layer

    """
    ### TYPE HERE AND REMOVE `pass` below ###
    sigmoid_layer = Sigmoid()
    #s_output = 1/(1+np.exp(-x))
    s_output = sigmoid_layer.forward(x)
    #da = np.multiply(np.multiply(z,(1-z)),self.g)
    s_grad_input = sigmoid_layer.backward(grad_higher)
    #s_grad_input = np.multiply(np.multiply(x,(1-x)),grad_higher)
    sigmoid_output = {'output':s_output,'grad_input':s_grad_input}
    return sigmoid_output


def compute_crossentropy(score: np.ndarray,
                         label: np.ndarray):

    """
    :param score: np.ndarray, shape (N, C), the score values for input of
                input of softmax ($b_k$ in the writeup).
                notice that here C can be other than 10.
    :param label: np.ndarray of integers, shape (N,), each value in [0,C-1]
                (non-zero idx of $\vec{y}$ in the writeup).

    :return: should be a dictionary containing following keys:

        'output': int, the mean negative cross entropy loss

        'grad_input': np.ndarray, shape (N, C), the gradient of loss
                      w.r.t. the score
    """
    #sum_score = np.sum(np.exp(score),axis=1)
    # y_hat = np.exp(score)/sum_score
    #e_output = (-np.sum(np.multiply(label,np.log(y_hat)),axis=1))*1/N
    #dyhat = -(1/N)*(label/y_hat)
    #dyhat_db = np.multiply(y_hat,(1-y_hat))
    #e_grad_input = np.multiply(dyhat_db,dyhat)
    crossentropy_layer = CrossEntropyLoss()
    e_output = crossentropy_layer.forward(score,label)
    e_grad_input = crossentropy_layer.backward()
    entropy_output = {'output':e_output,'grad_input':e_grad_input}
    return entropy_output


class NN(Module):
    def __init__(self,
                 weight_l1, bias_l1,
                 weight_l2, bias_l2,
                 ):
        super().__init__()
        self._register_child('lc_1',Linear(weight_l1,bias_l1))
        self._register_child('z',Sigmoid())
        self._register_child('lc_2',Linear(weight_l2,bias_l2))
        self._register_child('loss',CrossEntropyLoss())

    def forward(self, x, y):
        """
        :param x: np.ndarray, shape (N, in_features), training input
        :param y: np.ndarray, shape (N, ), training label

        :return: the mean negative cross entropy loss
        """
        fa = self.children['lc_1'].forward(x)
        fz = self.children['z'].forward(fa)
        fb = self.children['lc_2'].forward(fz)
        fl = self.children['loss'].forward(fb,y)
        return fl

    def backward(self):
        bl = self.children['loss'].backward()
        bb = self.children['lc_2'].backward(bl)
        bz = self.children['z'].backward(bb)
        ba = self.children['lc_1'].backward(bz)
        return ba


def define_network(init_scheme,
                   num_hidden,
                   seed=0, size_input=784,
                   size_output=10
                   ):
    """
    :param init_scheme: can be 'random' or 'zero', used to initialize the
                        weight matrices
    :param num_hidden: the number of hidden units
    :param seed: seed used to generate initial random weights. can be ignored.
    :param size_input: number of input features. can be ignored (784 for this
           assignment)
    :param size_output: number of output classes. can be ignored (10 for this
           assignment)
    """

    rng_state = np.random.RandomState(seed=seed)

    # generate weights according to init_scheme
    if(init_scheme=='zero'):
        weight_l1 = np.zeros([num_hidden,size_input])
        weight_l2 = np.zeros([size_output,num_hidden])
    if(init_scheme =='random'):
        weight_l1 = rng_state.rand(num_hidden,size_input)
        weight_l2 = rng_state.rand(size_output,num_hidden)

    # generate bias
    bias_l1 = np.zeros(num_hidden)
    bias_l2 = np.zeros(size_output)

    # generate network
    newnet = NN(weight_l1,bias_l1,weight_l2,bias_l2)
    return newnet


def clear_grad(nn):
    # clear grad info in the parameters.
    # hint: use nn.named_parameters, and the fact that given
    # an array x, x[...] = 0 empties it in place.
    """
    clear the gradient in the parameters and replace them with 0's
    """
    for name, param, grad in nn.named_parameters():
        grad[...] = 0


def update_param(nn, lr):
    # update parameters
    # hint: use nn.named_parameters, and the fact that given
    # two arrays x, y of same shape,
    # x += y updates x to be x + y in place.
    """
    update the parameters of the network
    """
    for name, param, grad in nn.named_parameters():
        param -= lr*grad


def train_network(nn, dataset, num_epoch,
                  learning_rate,
                  seed=0, no_shuffle=True, ):
    """
    :param nn: neural network object
    :param dataset: a dictionary of following keys:
            'train_x': np.ndarray, shape (N, 784)
            'train_y': np.ndarray, shape (N, )
            'test_x': np.ndarray of int, shape (N_test, 784)
            'test_y': np.ndarray of int, shape (N_test, )

            for training_data_student, we should have
            N=3000, and N_test=1000

    :param num_epoch: (E) the number of epochs to train the network
    :param learning_rate: the learning_rate multiplied on the gradients
    :param seed: an integer used to generate random initial weights,
           not needed for autolab.
    :param no_shuffle: a boolean indicating if the training dataset will be
                shuffled before training,
           keep default value for autolab.

    :return: should be a dictionary containing following keys:

        'train_loss': list of training losses, its size should equal E


        'test_error': list of testing losses, its size should equal E

        'train_error': list of training errors, its size should equal E

        'test_error': list of testing errors, its size should equal E

        'yhat_train': list of prediction labels for training dataset,
                      its size should equal N #3000

        'yhat_test': list of prediction labels for testing dataset,
                     its size should equal N_test #1000
    """
    # get data
    train_x, train_y = dataset['train_x'], dataset['train_y']
    test_x, test_y = dataset['test_x'], dataset['test_y']

    rng_state = np.random.RandomState(seed=seed)

    training_loss_all = []
    training_error_all = []
    testing_loss_all = []
    testing_error_all = []
    

    for idx_epoch in range(num_epoch):  # for each epoch
        
        if no_shuffle:
            # this is for autolab.
            shulffe_idx = np.arange(train_y.size)
        else:
            # this is for empirical questions.
            shulffe_idx = rng_state.permutation(train_y.size)

        for idx_example in shulffe_idx:  # for each training sample.
            x_this = train_x[idx_example:idx_example + 1]
            y_this = train_y[idx_example:idx_example + 1]


            # clear grad
            clear_grad(nn)

            # forward
            nn.forward(x_this,y_this)

            # generate grad
            nn.backward()

            # update parameters
            update_param(nn, learning_rate)
            

        # training_loss_this_epoch is average loss over training data.
        # note that you should compute this loss ONLY using the model
        nn.clear_cache()
        training_loss_this_epoch = nn.forward(train_x, train_y)

        # record training loss
        # this float() is just there so that result can be JSONified easily
        training_loss_all.append(float(training_loss_this_epoch))

        # generate predicted labels for training data
        # yhat_train_all should be a 1d vector of same shape as train_y
        
        yhat_train = nn.children['loss'].cache['y_hat']
        yhat_train_all = np.argmax(yhat_train,axis=1).reshape(len(train_y),)
            
        # record training error
        training_error_all.append(float((yhat_train_all != train_y).mean()))

        # training_loss_this_epoch is average loss over training data.
        nn.clear_cache()
        testing_loss_this_epoch = nn.forward(test_x,test_y)

        # record testing loss
        testing_loss_all.append(float(testing_loss_this_epoch))

        # generate yhat for testing data
        yhat_test = nn.children['loss'].cache['y_hat']
        yhat_test_all = np.argmax(yhat_test,axis=1).reshape(len(test_y),)
     
        # record testing error
        testing_error_all.append(float((yhat_test_all != test_y).mean()))

    # keep this part intact, do not modify it.
    return {
        # losses and errors across epochs.
        'train_loss': training_loss_all,
        'test_loss': testing_loss_all,
        'train_error': training_error_all,
        'test_error': testing_error_all,

        # yhat of the final model at the last epoch.
        # tolist for JSON.
        'yhat_train': yhat_train_all.tolist(),
        'yhat_test': yhat_test_all.tolist(),
    }


def autolab_trainer(dataset, init_scheme, num_hidden,
                    num_epoch, learning_rate, size_input=784):
    """

    :param dataset: dataset as in that of `train_network`
    :param init_scheme: init_scheme as in that of `define_network`
    :param num_hidden: num_hidden as in that of `define_network`
    :param num_epoch: num_epoch as in that of `train_network`
    :param learning_rate: num_epoch as in that of `train_network`
    :param size_input: size_input as in that of `define_network`,
           can be ignored fir this assignment
    :return: return value of `train_network`.
    """
    NN = define_network(init_scheme,num_hidden,seed=0, size_input=784,size_output=10)
    #dataset = load_data()
    final_output = train_network(NN, dataset, num_epoch,learning_rate,seed=0, no_shuffle=True,)
    return final_output
  


