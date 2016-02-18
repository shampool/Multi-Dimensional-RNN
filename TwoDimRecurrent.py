# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
import numpy as np

from .. import backend as K
from .. import activations, initializations
#from ..layers.core import MaskedLayer
from theano import tensor as T
import theano
from ..layers.recurrent import Recurrent
from ..layers.core import MaskedLayer 


import numpy as np

from collections import OrderedDict
import copy
from six.moves import zip

from .. import backend as K
from .. import activations, initializations, regularizers, constraints
from ..regularizers import ActivityRegularizer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import marshal
import types
import sys



class TwoDimRecurrent(MaskedLayer):
  
    input_ndim = 4

    def __init__(self, weights=None, droprate = None,
                 return_sequences=False, go_backwards=False, stateful=False,
                 input_dim=None, input_length=(None,None), truncate_gradient = -1,**kwargs):

        self.return_sequences = return_sequences         
        self.initial_weights = weights
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.truncate_gradient = truncate_gradient
        #self.single_state_num = 1
        self.input_dim = input_dim
        self.input_length = input_length
        self.droprate = droprate
        if self.input_dim:
            kwargs['input_shape'] = self.input_length + (self.input_dim,)
        super(TwoDimRecurrent, self).__init__(**kwargs)

    def get_output_mask(self, train=False):
        if self.return_sequences:
            return super(TwoDimRecurrent, self).get_output_mask(train)
        else:
            return None

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.return_sequences:
            return (input_shape[0], input_shape[1],input_shape[2], self.output_dim)
        else:
            return (input_shape[0], self.output_dim)

    def step(self, x, states,**kwargsdic):
        raise NotImplementedError



    def get_output(self, train=False):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        X = self.get_input(train)
        assert K.ndim(X) == 4
  
        #inner_data = K.reshape(X, (-1,) + self.grid_shape + (self.input_shape[2],))  # for example 16*16 *900 (16 * 16 grids and each contains 900 pixels)
        #inner_data =X  

        seed = np.random.randint(10e6)
        self.srng = RandomStreams(seed=seed)  #RandomStreams(seed=seed)
        DropOut_Matrix = self.get_dropout(batchsize = X.shape[0])

        if K._BACKEND == 'tensorflow':
            if not self.input_shape[1]:
                raise Exception('When using TensorFlow, you should define ' +
                                'explicitly the number of timesteps of ' +
                                'your sequences. Make sure the first layer ' +
                                'has a "batch_input_shape" argument ' +
                                'including the samples axis.')
    
        mask = self.get_output_mask(train)

        if mask:
            # apply mask
            X *= K.cast(K.expand_dims(mask), X.dtype)
            masking = True
        else:
            masking = False
     
        #grid_mask = K.reshape(mask, (-1,) +  self.grid_shape)
        
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X) #build an list of all-zero tensor of row_states, col_states
        
        
        
        
        last_output, outputs, states = self.rnn_grid(self.step, X ,initial_states,
                                             go_backwards=self.go_backwards, dropout_matrix = DropOut_Matrix, train = train,
                                             masking=masking)
        
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))
        
        if self.return_sequences:
            return outputs
        else:
            return last_output

    def rnn_grid(self ,step_function, inputs,initial_states,dropout_matrix = [],train=True,
        go_backwards=[False, False], masking=True):
        '''initial_states should be a list of tensor variable, 
        [row_tensor_state_1,...,row_tensor_state_n, col_tensor_state_1,..., col_tensor_state_n]
        '''
        #column first for all the starting corner, and go_backwards is [row_goback, col_back]
        #assert go_backwards not in ['leftupper','leftlower','rightupper','rightlower'] , "non valid go_backwards"
        if type(go_backwards) != 'list':
            go_backwards = [go_backwards]
        if len(go_backwards) == 2:
            [row_go_backwards, col_go_backwards] = go_backwards
        else:
            row_go_backwards = col_go_backwards = go_backwards
            
        grid_shape = (inputs.shape[1], inputs.shape[2])
        if len(initial_states) == 2:
           row_states = initial_states[0]  #now a list of tensor of shape, X.rowshape, dim
           col_states = initial_states[1]  #now a list  of tensor of shape, X.colshape, dim
        elif len(initial_states) == 1:
           row_states = col_states = initial_states[0]  #now a list of tensor of shape, X.rowshape, dim 
        elif len(initial_states) == 0:
            row_states = col_states = []  #now a list of tensor of shape, X.rowshape, dim
        else:
            raise Exception("Error length of parameters")   
        inputs = inputs.dimshuffle((2,1, 0, 3))  # col, row, nsample, dim
        #inputs = input.dimshuffle(tuple(range(1,len(grid_shape)+1))+(0, len(grid_shape)+1) )
                
        def _step(input,row, row_state, col_state, col): #[row_states] + [col_state]
            
            kwargsdic = {'row':row, 'col':col, 'dropout_matrix':dropout_matrix, 'train':train}
            output, new_state = step_function(input, row_state, col_state, **kwargsdic)  # this function actually should only return one state
    
            if masking:
    
                switch = T.any(input, axis=-1, keepdims=True)
                output = T.switch(switch, output, 0. * output)
                
                switch_ = T.shape_padright(switch)  # (nb_samples, 1, 1)
                switch_ = T.addbroadcast(switch_, -1)
        
                return_state = T.switch(switch_, new_state, col_state)
                
                return [output] + [return_state]
            else:
                return [output] + [new_state]
           
        def loop_over_col(coldata, col_ind, col_state,rows_states,rows_model):
    
            results , _ = theano.scan( fn = _step,
                                       sequences = [coldata, rows_model,rows_states],                                      
                                       outputs_info=[None] +  [col_state] ,
                                       non_sequences = [col_ind], 
                                       go_backwards = row_go_backwards,
                                       truncate_gradient = self.truncate_gradient
                                       )
                # deal with Theano API inconsistency
            if type(results) is list:            
                col_vals = results[0] 
                new_row_states = results[1]  
                if row_go_backwards == True:
                   col_vals = col_vals[::-1]  
                   new_row_states = new_row_states[::-1]
                   
                new_col_state = new_row_states[-1,::] 
                
                returned_row_states = [new_row_states]
                returned_col_state = [new_col_state]
                   
            else:
                col_vals = results      
                new_row_states = []
                if row_go_backwards == True:
                   col_vals = col_vals[::-1]  
                returned_row_states  = [] 
                returned_col_state  = []
  
            return [col_vals] + returned_row_states+  returned_col_state
        inputs_model = dict(input = inputs, taps=[0])
        rows_model = T.arange(grid_shape[0])
        cols_model = T.arange(grid_shape[1])

    
        results, _ = theano.scan( fn = loop_over_col,
                                    sequences = [inputs_model,cols_model,col_states],
                                    #'''return grid_results, row_states, col_state'''
                                    outputs_info=[None]+ [row_states] + [None], 
                                    non_sequences = [rows_model], 
                                    go_backwards = col_go_backwards,
                                    truncate_gradient = self.truncate_gradient 
            )
        if type(results) is list:
           outputs =  results[0]  #'''tensor type'''
           row_states_tensor  = results[1][-1,::]  #'''we only need the last column as the row_states, tensor type'''
           col_states_tensor  = results[2]  #'''tensor type'''
           retuned_row_states = [row_states_tensor]
           retuned_col_states = [col_states_tensor]
        else:
           outputs = results
           #row_states_tensor = []
           #col_states_tensor = []
           retuned_row_states = []
           retuned_col_states = [] 
        #outputs =  results[0]  #'''tensor type'''
        #row_states_tensor  = results[1][-1,::]  #'''we only need the last column as the row_states, tensor type'''
        #col_states_tensor  = results[2]  #'''tensor type'''
    
        outputs = T.squeeze(outputs)
        last_output = outputs[-1,-1,::]
    
        #outputs  col, row, nsample, dim
        if col_go_backwards == True:
           outputs = outputs[::-1]
           col_states_tensor = col_states_tensor[::-1]
           retuned_col_states = [col_states_tensor]
        outputs = outputs.dimshuffle((2,1,0,3))
              
        return last_output, outputs, retuned_row_states + retuned_col_states


    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length,
                  "go_backwards": self.go_backwards,
                  "stateful": self.stateful}
        base_config = super(TwoDimRecurrent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


        
        
class TwoDimRNN(TwoDimRecurrent):

    def __init__(self, output_dim ,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 **kwargs):
        
        self.output_dim = output_dim
        #self.grid_shape = tuple(grid_shape)
        self.single_state_num = 1
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        
        #self.input = K.placeholder(ndim=4)
        
        super(TwoDimRNN, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None, None] # for _ in range(np.sum(self.grid_shape))]
            #for state_id in range(np.sum(self.grid_shape)):
            #  self.states.append(K.zeros((input_shape[0], self.output_dim)) )

        input_dim = input_shape[3] #Nsample*timestep*input_dim
        self.input_dim = input_dim

        self.W = self.init((input_dim, self.output_dim))
        self.U_row = self.inner_init((self.output_dim, self.output_dim))
        self.U_col = self.inner_init((self.output_dim, self.output_dim))
        self.b = K.zeros((self.output_dim,))
        self.params = [self.W, self.U_row, self.U_col, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
            
    def get_initial_states(self, X):
        #initial_states should be a list of tensor variable, 
        # [row_tensor_states, col_tensor_states], of size row_Step * Nsample, *output_dim * single_state_num, 
        #and col_Step * Nsample, *input_dim * single_state_num, respectively
        
        initial_state = K.zeros_like(X)  # (samples, row_timesteps,col_timesteps, input_dim)
        
        initial_state = initial_state.dimshuffle((1,2, 0, 3)) #(row_timesteps,col_timesteps, samples, input_dim)     
        initial_state_row = K.sum(initial_state, axis=1)  # (samples, row_timesteps, input_dim)
        initial_state_col = K.sum(initial_state, axis=0)  # (samples, col_timesteps, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        
        #initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_state_row_list = [K.dot(initial_state_row, reducer) for _ in range(self.single_state_num)]
        initial_state_col_list = [K.dot(initial_state_col, reducer)  for _ in range(self.single_state_num)]
        
        initial_states = [T.stack(initial_state_row_list, axis=-1), T.stack(initial_state_col_list, axis=-1)]
        return initial_states
        
    def reset_states(self):  #used in the model building stage
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[1], input_shape[0], self.output_dim, self.single_state_num)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[2], input_shape[0], self.output_dim, self.single_state_num)))

        else:
            self.states = []
            for state_id in range(np.sum(self.grid_shape)):
                self.states.append(K.zeros((input_shape[0], self.output_dim)))

    def get_dropout(self,batchsize= None):
        #if self.seed is None:   
        if self.droprate is not None:
          if len(self.droprate) == 1:
             self.droprate.append(self.droprate[0])  
          retain_prob_W = 1- self.droprate[0]
          retain_prob_U = 1- self.droprate[1]
          B_W = self.srng.bionomial((1, batchsize, self.input_dim), p = retain_prob_W, dtype = theano.config.floatX)
          B_U = self.srng.bionomial((2, batchsize, self.output_dim), p = retain_prob_U, dtype = theano.config.floatX)
          return [B_W, B_U] 
        else:
          return []

    def step(self, x, row_state, col_state,**kwargsdic):
        # states only contains the previous output.
        #assert len(states) == 1
        #prev_output = states[0]
        dropout_matrix = kwargsdic['dropout_matrix']
        train = kwargsdic['train']
        r_state_1 = row_state[:,:,0]
        c_state_1 = col_state[:,:,0]
        
        if len(dropout_matrix) != 0:
            if train:
                x = x*dropout_matrix[0][0]/self.droprate[0]
                r_state_1 = r_state_1 *dropout_matrix[1][0] /self.droprate[1]
                c_state_1 = c_state_1 *dropout_matrix[1][1] /self.droprate[1]
                
        h = K.dot(x , self.W) + self.b
        output = self.activation(h + (K.dot(r_state_1, self.U_row) + K.dot(c_state_1, self.U_col)) )
        
        return output, T.stack([output], axis =-1) 
         

    
    def get_config(self):
        config = {
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__}
        base_config = super(TwoDimRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TwoDimGRU(TwoDimRecurrent):

    def __init__(self, output_dim , 
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 **kwargs):
        
        self.output_dim = output_dim
        #self.grid_shape = tuple(grid_shape)
        self.single_state_num = 1
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
 
        super(TwoDimGRU, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        if self.stateful:
            self.reset_states()
        else:
            self.states = [None, None] # one for row and one for col state

        input_dim = input_shape[3] #Nsample*timestep*input_dim
        self.input_dim = input_dim


        self.W_z = self.init((input_dim, self.output_dim))        
        self.b_z_row = K.variable(np.ones((self.output_dim,)) * -5) 
        self.b_z_col = K.variable(np.ones((self.output_dim,)) * -5)
        self.U_z_row = self.inner_init((self.output_dim, self.output_dim))
        self.U_z_col = self.inner_init((self.output_dim, self.output_dim))
        
        self.W_r = self.init((input_dim, self.output_dim))        
        self.b_r_row = K.variable(np.ones((self.output_dim,)) * -5)
        self.b_r_col = K.variable(np.ones((self.output_dim,)) * -5)
        self.U_r_row = self.inner_init((self.output_dim, self.output_dim))
        self.U_r_col = self.inner_init((self.output_dim, self.output_dim))
        
        self.W_h = self.init((input_dim, self.output_dim))        
        self.b_h = K.zeros((self.output_dim,))
        self.U_h_row = self.inner_init((self.output_dim, self.output_dim))
        self.U_h_col = self.inner_init((self.output_dim, self.output_dim))
        

        self.params = [self.W_z, self.U_z_row, self.U_z_col, self.b_z_row, self.b_z_col,
                       self.W_r, self.U_r_row, self.U_r_col, self.b_r_row, self.b_r_col,
                       self.W_h, self.U_h_row, self.U_h_col,self.b_h]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
            
    def get_initial_states(self, X):
        #initial_states should be a list of tensor variable, 
        # [row_tensor_states, col_tensor_states], of size row_Step * Nsample, *output_dim * single_state_num, 
        #and col_Step * Nsample, *input_dim * single_state_num, respectively
        
        initial_state = K.zeros_like(X)  # (samples, row_timesteps,col_timesteps, input_dim)
        
        initial_state = initial_state.dimshuffle((1,2, 0, 3)) #(row_timesteps,col_timesteps, samples, input_dim)     
        initial_state_row = K.sum(initial_state, axis=1)  # (row_timesteps, samples,  input_dim)
        initial_state_col = K.sum(initial_state, axis=0)  # (col_timesteps, samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        
        #initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_state_row_list = [K.dot(initial_state_row, reducer) for _ in range(self.single_state_num)]
        initial_state_col_list = [K.dot(initial_state_col, reducer)  for _ in range(self.single_state_num)]
        
        initial_states = [T.stack(initial_state_row_list, axis=-1), T.stack(initial_state_col_list, axis=-1)]
        return initial_states
        
    def reset_states(self):  #used in the model building stage
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[1], input_shape[0], self.output_dim, self.single_state_num)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[2], input_shape[0], self.output_dim, self.single_state_num)))

        else:
            self.states = []
            for state_id in range(np.sum(self.grid_shape)):
                self.states.append(K.zeros((input_shape[0], self.output_dim)))

    def get_dropout(self,batchsize = None):

        #seed = np.random.randint(10e6)
        #self.srng = RandomStreams(seed=seed)
        if self.droprate is not None:
          if len(self.droprate) == 1:
             self.droprate = self.droprate*2

          retain_prob_W = 1- self.droprate[0]
          retain_prob_U = 1- self.droprate[1]
          B_W = self.srng.binomial((3, batchsize, self.input_dim), p = retain_prob_W, dtype = theano.config.floatX)
          B_U = self.srng.binomial((6, batchsize, self.output_dim), p = retain_prob_U, dtype = theano.config.floatX)
          
          return [B_W, B_U] 
        else:
          return []
  
    def step(self, x, row_state, col_state,**kwargsdic):
        
        r_state_1 = row_state[:,:,0]
        c_state_1 = col_state[:,:,0]
        dropout_matrix = kwargsdic['dropout_matrix']    
        train = kwargsdic['train']    
        
        if len(dropout_matrix) != 0:
            if train:
                x_z = K.dot(x * dropout_matrix[0][0], self.W_z)/self.droprate[0]
                x_r = K.dot(x * dropout_matrix[0][1], self.W_r) /self.droprate[0]
                x_h = K.dot(x * dropout_matrix[0][2], self.W_h) /self.droprate[0]
                
                r_hh = (K.dot(r_state_1 * dropout_matrix[1][0], self.U_r_row) \
                  + K.dot(c_state_1 * dropout_matrix[1][1], self.U_r_col) ) /self.droprate[1]
                  
                r_row = self.inner_activation(x_r + r_hh + self.b_r_row) 
                r_col = self.inner_activation(x_r + r_hh + self.b_r_col)
                
                z_hh = (K.dot(r_row *r_state_1 * dropout_matrix[1][2], self.U_h_row) \
                  + K.dot(r_col *c_state_1 * dropout_matrix[1][3], self.U_h_col) ) /self.droprate[1]
                
                z_row_sum = K.dot(r_state_1 * dropout_matrix[1][4], self.U_z_row)/self.droprate[1]
                z_col_sum = K.dot(c_state_1 * dropout_matrix[1][5], self.U_z_col)/self.droprate[1]
                
                z_row = self.inner_activation(x_z + z_row_sum  + self.b_z_row )
                z_col = self.inner_activation(x_z +  z_col_sum + self.b_z_col )
            
            else:
                x_z = K.dot(x, self.W_z)
                x_r = K.dot(x, self.W_r) 
                x_h = K.dot(x, self.W_h) 
                        
                r_hh = (K.dot(r_state_1, self.U_r_row) + K.dot(c_state_1, self.U_r_col) )
                r_row = self.inner_activation(x_r + r_hh + self.b_r_row)
                r_col = self.inner_activation(x_r + r_hh + self.b_r_col)
                
                z_hh = (K.dot(r_row *r_state_1, self.U_h_row) + K.dot(r_col *c_state_1, self.U_h_col) )
                
                z_row_sum = K.dot(r_state_1 , self.U_z_row)
                z_col_sum = K.dot(c_state_1 , self.U_z_col)
                
        else:
                x_z = K.dot(x, self.W_z)
                x_r = K.dot(x, self.W_r) 
                x_h = K.dot(x, self.W_h) 
                        
                r_hh = (K.dot(r_state_1, self.U_r_row) + K.dot(c_state_1, self.U_r_col) )
                r_row = self.inner_activation(x_r + r_hh + self.b_r_row)
                r_col = self.inner_activation(x_r + r_hh + self.b_r_col)
                
                z_hh = (K.dot(r_row *r_state_1, self.U_h_row) + K.dot(r_col *c_state_1, self.U_h_col) )
                
                z_row_sum = K.dot(r_state_1 , self.U_z_row)
                z_col_sum = K.dot(c_state_1 , self.U_z_col)        
                
        hh = self.activation(x_h  + z_hh + self.b_h)                
        z_row = self.inner_activation(x_z + z_row_sum  + self.b_z_row )
        z_col = self.inner_activation(x_z +  z_col_sum + self.b_z_col )   
        h = (1- z_col - z_row) *hh + z_row*r_state_1 +  z_col * c_state_1

        return h, T.stack([h], axis =-1)

         

    def get_config(self):
        config = {
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__}
        base_config = super(TwoDimGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class TwoDimLSTM(TwoDimRecurrent):

    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid', **kwargs):
        
        self.output_dim = output_dim
        #self.grid_shape = tuple(grid_shape)
        self.single_state_num = 2
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
 
        super(TwoDimLSTM, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        if self.stateful:
            self.reset_states()
        else:
            self.states = [None, None] # one for row and one for col state

        input_dim = input_shape[3] #Nsample*timestep*input_dim
        self.input_dim = input_dim


        self.W_i = self.init((input_dim, self.output_dim))
        self.U_i_hid_row = self.inner_init((self.output_dim, self.output_dim))
        self.U_i_hid_col = self.inner_init((self.output_dim, self.output_dim)) 
        self.U_i_cel_row = self.inner_init((self.output_dim, self.output_dim)) 
        self.U_i_cel_col = self.inner_init((self.output_dim, self.output_dim)) 
        self.b_i = K.zeros((self.output_dim,))

        self.W_f = self.init((input_dim, self.output_dim))
        self.U_f_hid_row = self.inner_init((self.output_dim, self.output_dim))
        self.U_f_hid_col = self.inner_init((self.output_dim, self.output_dim))
        self.U_f_cel_row = self.inner_init((self.output_dim, self.output_dim)) 
        self.U_f_cel_col = self.inner_init((self.output_dim, self.output_dim)) 
        self.b_f_row = K.variable(np.ones((self.output_dim,)) * -5) #self.forget_bias_init((self.output_dim)) * (-5)
        self.b_f_col = K.variable(np.ones((self.output_dim,)) * -5) #self.forget_bias_init((self.output_dim)) * (-5)
        
        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c_hid_row = self.inner_init((self.output_dim, self.output_dim))
        self.U_c_hid_col = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = K.zeros((self.output_dim,))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o_hid_row = self.inner_init((self.output_dim, self.output_dim))
        self.U_o_hid_col = self.inner_init((self.output_dim, self.output_dim))
        self.U_c_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = K.zeros((self.output_dim,))
       

        #self.W_x = T.as_tensor_variable(K.concatenate([self.W_i, self.W_f,self.W_c, self.W_o], axis = -1))
        
        #self.U_hid_row = T.as_tensor_variable(K.concatenate([self.U_i_hid_row, self.U_f_hid_row,self.U_c_hid_row, self.U_o_hid_row], axis = -1))
        #self.U_hid_col = T.as_tensor_variable(K.concatenate([self.U_i_hid_col, self.U_f_hid_col,self.U_c_hid_col, self.U_o_hid_col], axis = -1))
        #self.U_cel_row = T.as_tensor_variable(K.concatenate([self.U_i_cel_row, self.U_f_cel_row], axis = -1))
        #self.U_cel_col = T.as_tensor_variable((K.concatenate([self.U_i_cel_col, self.U_f_cel_col], axis = -1)))
        
        
        
#        self.params = [self.W_x, 
#                       self.U_hid_row, self.U_hid_col, self.U_cel_row, self.U_cel_col,
#                       self.b_i,self.b_f_row,self.b_f_col,self.b_c,
#                       self.U_c_o, self.b_o]
        self.params = [self.W_i, self.W_f,self.W_c, self.W_o, 
                       self.U_i_hid_row, self.U_f_hid_row,self.U_c_hid_row, self.U_o_hid_row,
                       self.U_i_hid_col, self.U_f_hid_col,self.U_c_hid_col, self.U_o_hid_col,
                       self.U_i_cel_row, self.U_f_cel_row,
                       self.U_i_cel_col, self.U_f_cel_col,
                       self.b_i,self.b_f_row,self.b_f_col,self.b_c,
                       self.U_c_o, self.b_o]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
            
    def get_initial_states(self, X):
        #initial_states should be a list of tensor variable, 
        # [row_tensor_states, col_tensor_states], of size row_Step * Nsample, *output_dim * single_state_num, 
        #and col_Step * Nsample, *input_dim * single_state_num, respectively
        
        initial_state = K.zeros_like(X)  # (samples, row_timesteps,col_timesteps, input_dim)
        
        initial_state = initial_state.dimshuffle((1,2, 0, 3)) #(row_timesteps,col_timesteps, samples, input_dim)     
        initial_state_row = K.sum(initial_state, axis=1)  # (row_timesteps, samples,  input_dim)
        initial_state_col = K.sum(initial_state, axis=0)  # (col_timesteps, samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        
        #initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_state_row_list = [K.dot(initial_state_row, reducer) for _ in range(self.single_state_num)]
        initial_state_col_list = [K.dot(initial_state_col, reducer)  for _ in range(self.single_state_num)]
        
        initial_states = [T.stack(initial_state_row_list, axis=-1), T.stack(initial_state_col_list, axis=-1)]
        return initial_states
        
    def reset_states(self):  #used in the model building stage
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[1], input_shape[0], self.output_dim, self.single_state_num)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[2], input_shape[0], self.output_dim, self.single_state_num)))

        else:
            self.states = []
            for state_id in range(np.sum(self.grid_shape)):
                self.states.append(K.zeros((input_shape[0], self.output_dim)))
    def get_dropout(self,batchsize = None):

        #seed = np.random.randint(10e6)
        #self.srng = RandomStreams(seed=seed)
        if self.droprate is not None:
          if len(self.droprate) == 1:
             self.droprate = self.droprate*2

          retain_prob_W = 1- self.droprate[0]
          retain_prob_U = 1- self.droprate[1]
          B_W = self.srng.binomial((4,  batchsize, self.input_dim), p = retain_prob_W, dtype = theano.config.floatX)
          B_U = self.srng.binomial((12, batchsize, self.output_dim), p = retain_prob_U, dtype = theano.config.floatX)
          
          return [B_W, B_U] 
        else:
          return []

    def step(self, x, row_state, col_state,**kwargsdic):

                       
        def _slice(_x, n, dim):
            return _x[:,n * dim:(n + 1) * dim]
     
        r_state_hid = row_state[:,:,0]
        r_state_cel = row_state[:,:,1]

        c_state_hid = col_state[:,:,0]
        c_state_cel = col_state[:,:,1]
        dropout_matrix = kwargsdic['dropout_matrix']    
        if len(dropout_matrix) == 0:
            #X_X = K.dot(x,self.W_x) # W_X is W_i + W_f + W_c + W_o
            #x_i = _slice(X_X, 0, self.output_dim)
            #x_f = _slice(X_X, 1, self.output_dim)
            #x_c = _slice(X_X, 2, self.output_dim)
            #x_o = _slice(X_X, 3, self.output_dim)
            
            
            # U_hid_row is U_i_hid_row, U_f_hid_row,U_c_hid_row, U_o_hid_row
            # U_hid_col is U_i_hid_col, U_f_hid_col,U_c_hid_col, U_o_hid_col
            
            #ROW_HID_U = K.dot(r_state_hid, self.U_hid_row)
            #COL_HID_U = K.dot(c_state_hid, self.U_hid_col)
            
            # U_cel_row is U_i_cel_row, U_f_cel_row
            # U_cel_col is U_i_cel_col, U_f_cel_col
            #ROW_CEL_U = K.dot(r_state_cel, self.U_cel_row)
            #COL_CEL_U = K.dot(c_state_cel, self.U_cel_col)


            x_i = K.dot(x , self.W_i) 
            hid_i = (K.dot(r_state_hid, self.U_i_hid_row) + K.dot(c_state_hid, self.U_i_hid_col) )
            cel_i = (K.dot(r_state_cel, self.U_i_cel_row) + K.dot(c_state_cel, self.U_i_cel_col) )
            #hid_i = _slice(ROW_HID_U, 0, self.output_dim) + _slice(COL_HID_U, 0, self.output_dim)
            #cel_i = _slice(ROW_CEL_U, 0, self.output_dim) + _slice(COL_CEL_U, 0, self.output_dim)
            i_t = self.inner_activation(x_i+ self.b_i + hid_i + cel_i)
            
            x_f = K.dot(x , self.W_f) 
            hid_f_row = (K.dot(r_state_hid, self.U_f_hid_row) + K.dot(c_state_hid, self.U_f_hid_col) )
            cel_f_row = K.dot(r_state_cel, self.U_f_cel_row) 
            #hid_f_row = _slice(ROW_HID_U, 1, self.output_dim) + _slice(COL_HID_U, 1, self.output_dim)
            #cel_f_row = _slice(ROW_CEL_U, 1, self.output_dim) + _slice(COL_CEL_U, 1, self.output_dim)
            f_row_t = self.inner_activation(x_f +  self.b_f_row  + hid_f_row + cel_f_row)
            
            #hid_f_col = (K.dot(r_state_hid, self.U_f_hid_row) + K.dot(c_state_hid, self.U_f_hid_col) )
            cel_f_col = K.dot(c_state_cel, self.U_f_cel_col) 
            #hid_f_col = _slice(ROW_HID_U, 1, self.output_dim) + _slice(COL_HID_U, 1, self.output_dim)
            #cel_f_col = _slice(ROW_CEL_U, 1, self.output_dim) + _slice(COL_CEL_U, 1, self.output_dim)
            f_col_t = self.inner_activation(x_f + self.b_f_col + hid_f_row + cel_f_col)
            
            
            x_c = K.dot(x , self.W_c) 
            hid_c = (K.dot(r_state_hid, self.U_c_hid_row) + K.dot(c_state_hid, self.U_c_hid_col) )
            #hid_c = _slice(ROW_HID_U, 2, self.output_dim) + _slice(COL_HID_U, 2, self.output_dim)
            c_candidat_t = self.activation(x_c + self.b_c + hid_c)
            
           
            c_t = f_row_t * r_state_cel + f_col_t * c_state_cel + i_t*c_candidat_t
            
            x_o = K.dot(x , self.W_o) 
            o_h = (K.dot(r_state_hid, self.U_o_hid_row) + K.dot(c_state_hid, self.U_o_hid_col) )
            #o_h = _slice(ROW_HID_U, 3, self.output_dim) + _slice(COL_HID_U, 3, self.output_dim)
            o_t = self.inner_activation(x_o + self.b_o + o_h + K.dot(c_t, self.U_c_o)) #output gate

        else:

            x_i = K.dot(x * dropout_matrix[0][0], self.W_i) 
            hid_i = (K.dot(r_state_hid * dropout_matrix[1][0], self.U_i_hid_row) + K.dot(c_state_hid* dropout_matrix[1][1], self.U_i_hid_col) )
            cel_i = (K.dot(r_state_cel* dropout_matrix[1][2], self.U_i_cel_row) + K.dot(c_state_cel* dropout_matrix[1][3], self.U_i_cel_col) )
            i_t = self.inner_activation(x_i+ self.b_i + hid_i + cel_i)
            
            x_f = K.dot(x * dropout_matrix[0][1], self.W_f) 
            hid_f_row = (K.dot(r_state_hid * dropout_matrix[1][4], self.U_f_hid_row) + K.dot(c_state_hid* dropout_matrix[1][5], self.U_f_hid_col) )
            cel_f_row = K.dot(r_state_cel * dropout_matrix[1][6], self.U_f_cel_row) 
            f_row_t = self.inner_activation(x_f +  self.b_f_row  + hid_f_row + cel_f_row)            
            cel_f_col = K.dot(c_state_cel * dropout_matrix[1][7], self.U_f_cel_col) 
            f_col_t = self.inner_activation(x_f + self.b_f_col + hid_f_row + cel_f_col)
            
            
            x_c = K.dot(x * dropout_matrix[0][2], self.W_c) 
            hid_c = (K.dot(r_state_hid* dropout_matrix[1][8], self.U_c_hid_row) + K.dot(c_state_hid* dropout_matrix[1][9], self.U_c_hid_col) )       
            c_candidat_t = self.activation(x_c + self.b_c + hid_c)          
            c_t = f_row_t * r_state_cel + f_col_t * c_state_cel + i_t*c_candidat_t
            
            x_o = K.dot(x* dropout_matrix[0][3], self.W_o) 
            o_h = (K.dot(r_state_hid* dropout_matrix[1][10], self.U_o_hid_row) + K.dot(c_state_hid* dropout_matrix[1][11], self.U_o_hid_col) )
            o_t = self.inner_activation(x_o + self.b_o + o_h + K.dot(c_t, self.U_c_o)) #output gate
            
        h_t = o_t * self.activation(c_t)
       
        return h_t, T.stack([h_t, c_t], axis =-1)

         

    def get_config(self):
        config = {
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__}
        base_config = super(TwoDimLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

        
class TwoDimMasking(MaskedLayer):
    '''Mask an input sequence by using a mask value to identify padding.

    This layer copies the input to the output layer with identified padding
    replaced with 0s and creates an output mask in the process.

    At each timestep, if the values all equal `mask_value`,
    then the corresponding mask value for the timestep is 0 (skipped),
    otherwise it is 1.
    '''
    def __init__(self, mask_value=0., **kwargs):
        super(TwoDimMasking, self).__init__(**kwargs)
        self.mask_value = mask_value
        self.input = K.placeholder(ndim=4)

    def get_output_mask(self, train=False):
        if K._BACKEND == 'tensorflow':
            raise Exception('Masking is Theano-only for the time being.')
        X = self.get_input(train)
        return K.any(K.ones_like(X) * (1. - K.equal(X, self.mask_value)),
                     axis=-1)

    def get_output(self, train=False):
        X = self.get_input(train)

        return X * K.any((1. - K.equal(X, self.mask_value)),
                         axis=-1, keepdims=True)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'mask_value': self.mask_value}
        base_config = super(TwoDimMasking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))       


class TwoDimTimeDistributedDense(MaskedLayer):

    input_ndim = 4

    def __init__(self, output_dim, droprate = None,
                 init='glorot_uniform', activation='linear', weights=None,return_sequences = True,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.droprate = droprate
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        self.input = K.placeholder(ndim=4)
        super(TwoDimTimeDistributedDense, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[3]
        self.input_dim = input_dim
        self.W = self.init((input_dim, self.output_dim))
        self.b = K.zeros((self.output_dim,))

        self.params = [self.W, self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1], input_shape[2],self.output_dim)
        
    def rnn_grid_nostate(self ,step_function, inputs,
        go_backwards=[False, False], masking=True, dropout_matrix = [], train = True):
        '''initial_states should be a list of tensor variable, 
        [row_tensor_state_1,...,row_tensor_state_n, col_tensor_state_1,..., col_tensor_state_n]
        '''
        #column first for all the starting corner, and go_backwards is [row_goback, col_back]
        #assert go_backwards not in ['leftupper','leftlower','rightupper','rightlower'] , "non valid go_backwards"
        if type(go_backwards) != 'list':
            go_backwards = [go_backwards]
        if len(go_backwards) == 2:
            [row_go_backwards, col_go_backwards] = go_backwards
        else:
            row_go_backwards = col_go_backwards = go_backwards
            
        #grid_shape = (inputs.shape[1], inputs.shape[2])

        inputs = inputs.dimshuffle((2,1, 0, 3))  # col, row, nsample, dim
        #inputs = input.dimshuffle(tuple(range(1,len(grid_shape)+1))+(0, len(grid_shape)+1) )
                
        def _step(input): #[row_states] + [col_state]

            kwargsdic = {'dropout_matrix': dropout_matrix, 'train':train}
            
            output = step_function(input, **kwargsdic)  # this function actually should only return one state
    
            if masking:
    
                switch = T.any(input, axis=-1, keepdims=True)
                output = T.switch(switch, output, 0. * output)
                                   
                return [output] 
            else:
                return [output]
           
        def loop_over_col(coldata): 
    
            results , _ = theano.scan( fn = _step,
                                       sequences = [coldata],                                      
                                       outputs_info=[None],
                                       go_backwards = row_go_backwards,
                                       )
                # deal with Theano API inconsistency
            if type(results) is list:            
                col_vals = results[0] 
                if row_go_backwards == True:
                   col_vals = col_vals[::-1]                     
            else:
                col_vals = results      
                if row_go_backwards == True:
                   col_vals = col_vals[::-1]  

  
            return [col_vals]
        inputs_model = dict(input = inputs, taps=[0])
        #rows_model = T.arange(grid_shape[0])
        #cols_model = T.arange(grid_shape[1])    
        results, _ = theano.scan( fn = loop_over_col,
                                    sequences = [inputs_model],
                                    #'''return grid_results, row_states, col_state'''
                                    outputs_info=[None],  
                                    go_backwards = col_go_backwards,
            )
        if type(results) is list:
           outputs =  results[0]  #'''tensor type'''
        else:
           outputs = results
        outputs = T.squeeze(outputs)
        last_output = outputs[-1,-1,::]    
        #outputs  col, row, nsample, dim
        if col_go_backwards == True:
           outputs = outputs[::-1]
        outputs = outputs.dimshuffle((2,1,0,3))
        
        retuned_row_states = []
        retuned_col_states = []
           
        return last_output, outputs, retuned_row_states + retuned_col_states      
        
    def step(self, x, **kwargsdic):
         dropout_matrix = kwargsdic['dropout_matrix'] 
         train = kwargsdic['train'] 
         if len(dropout_matrix) != 0:
             if train:       
                x = x *dropout_matrix[0][0] / self.droprate[0]
        
         output = K.dot(x, self.W) + self.b
         return output        
                
#    def step(self, x, **kwargsdic):
#        dropout_matrix = kwargsdic['dropout_matrix'] 
#        train = kwargsdic['train']  
#        x = x *dropout_matrix[0][0] / self.droprate[0]
#        
#        output = K.dot(x, self.W) + self.b
#        return output        
                                
    
    def get_dropout(self,batchsize = None):

        #seed = np.random.randint(10e6)
        #self.srng = RandomStreams(seed=seed)
        if self.droprate is not None:

          retain_prob_W = 1- self.droprate[0]
          B_W = self.srng.binomial((1, batchsize, self.input_dim), p = retain_prob_W, dtype = theano.config.floatX)
                  
          return [B_W] 
        else:
          return []
          
    def get_output(self, train=False):
        X = self.get_input(train)
        seed = np.random.randint(10e6)
        self.srng = RandomStreams(seed=seed)
        DropOut_Matrix = self.get_dropout(batchsize = X.shape[0])

        last_output, outputs, _ = self.rnn_grid_nostate(self.step, X, masking=True, dropout_matrix = DropOut_Matrix, train= train)

        if self.return_sequences:
            return self.activation(outputs)
        else:
            return self.activation(last_output)
    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'input_dim': self.input_dim,
                  'input_length': self.input_length}
        base_config = super(TwoDimTimeDistributedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
