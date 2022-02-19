import torch
from torch import nn

class mlp(nn.Module):
    def __init__(self,list_of_layers):
        super(mlp,self).__init__()
        self.n_layers = len(list_of_layers)
        self.list_of_sequence = [None] * (4*self.n_layers)
        self.elements_in_sequence = 0
        for i in range(self.n_layers):
            self.add_layer_dict_to_sequence(**list_of_layers[i])
        self.model = nn.Sequential(*self.list_of_sequence[:self.elements_in_sequence])
        self.model.apply(init_weights)

    def add_layer_dict_to_sequence(self,**kwargs):
        self.add_element_to_sequence(nn.Linear(kwargs['n_in'],kwargs['n_out']))
        if kwargs['normalize']:
            self.add_element_to_sequence(nn.BatchNorm1d(kwargs['n_out'], 0.8))
        if kwargs['activation'] == 'sigmoid':
            self.add_element_to_sequence(nn.Sigmoid())
        if kwargs['activation'] == 'leakyrelu':
            self.add_element_to_sequence(nn.LeakyReLU(0.1, inplace=True))
        if kwargs['activation'] == 'tanh':
            self.add_element_to_sequence(nn.Tanh())
        if kwargs['dropout']:
            self.add_element_to_sequence(nn.Dropout(p=kwargs['dropout']))

    def add_element_to_sequence(self,element):
        self.list_of_sequence[self.elements_in_sequence] = element
        self.elements_in_sequence += 1

    def forward(self,input):
        return self.model(input)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def create_layer_dict(n_in,n_out,normalize=True,activation='leakyrelu',dropout=None):
    layer = {'n_in':n_in,'n_out':n_out}
    layer['normalize'] = normalize
    layer['activation'] = activation
    layer['dropout'] = dropout
    return layer
