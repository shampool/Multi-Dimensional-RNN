# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from .. import backend as K
from .. import activations, initializations
#from ..layers.core import MaskedLayer
from theano import tensor as T
import theano
from ..layers.recurrent import Recurrent
class CWRNN(Recurrent):
    '''ClockWork RNN .

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.

    # References
        
    '''
    
    def __init__(self, output_dim, periods,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 **kwargs):
        
        self.output_dim = output_dim
        self.periods = periods
        assert self.output_dim % len(self.periods) == 0
        
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        super(CWRNN, self).__init__(**kwargs)

    def build(self):
        nodes_period = self.output_dim // len(self.periods)
        mask = np.zeros((self.output_dim, self.output_dim))
        period = np.zeros((self.output_dim,), 'i')		
        for i, t_ in enumerate(self.periods):
           mask[i*nodes_period:(i+1)*nodes_period, i*nodes_period:] = 1
           period[i*nodes_period:(i+1)*nodes_period] = t_
        self._mask = K.variable(mask)
        self._period = K.variable(period)

        input_shape = self.input_shape
		
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None, None]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.W = self.init((input_dim, self.output_dim))
        self.U = self.inner_init((self.output_dim, self.output_dim))
        self.b = K.zeros((self.output_dim,))
        self.params = [self.W, self.U, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)), K.zeros((input_shape[0], self.output_dim))]

    def step(self, t, x, states):
        # states only contains the previous output.
        assert len(states) == 2
        prev_output = states[0]
        prev_p = states[1]
        
        x_t = K.dot(x, self.W) + self.b
        p = x_t + K.dot(prev_output, self.U*self._mask)
        
        p_t = K.switch(K.equal(t %self._period, 0), p, prev_p)
        output = self.activation(p_t)
        return output, [output, p_t]
    def get_output(self, train=False):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        X = self.get_input(train)
        assert K.ndim(X) == 3
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

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)
        
        last_output, outputs, states = K.rnn_time(self.step, X, initial_states,
                                             go_backwards=self.go_backwards,
                                             masking=masking)
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output
		
    def get_config(self):
        config = {"periods": self.periods,
		          "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__}
        base_config = super(CWRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MultiDimRNN(Recurrent):
    '''grid_shape: define the 2D shape of the grids. Eg. there are in total 5*6 grids'''

    
    def __init__(self, output_dim, grid_shape , 
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 **kwargs):
        
        self.output_dim = output_dim
        self.grid_shape = tuple(grid_shape)
  
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        super(MultiDimRNN, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None for _ in range(np.sum(self.grid_shape))]
            #for state_id in range(np.sum(self.grid_shape)):
            #    self.states.append(K.zeros((input_shape[0], self.output_dim)) )

        input_dim = input_shape[2] #Nsample*timestep*input_dim
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
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(X)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states
        
    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')
        if hasattr(self, 'states'):
            for state_id in range(np.sum(self.grid_shape)):  
                K.set_value(self.states[state_id],
                            np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = []
            for state_id in range(np.sum(self.grid_shape)):
                self.states.append(K.zeros((input_shape[0], self.output_dim)))

    def step(self, x, row_state, col_state):
        # states only contains the previous output.
        #assert len(states) == 1
        #prev_output = states[0]
        h = K.dot(x, self.W) + self.b
        output = self.activation(h + (K.dot(row_state, self.U_row) + K.dot(col_state, self.U_col))/2 )
        return output, [output]

    def get_output(self, train=False):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        X = self.get_input(train)
        assert K.ndim(X) == 3
        # get input, then map it to the grid_shape
        inner_data = K.reshape(X, (-1,) + self.grid_shape + (self.input_shape[2],))  # for example 16*16 *900 (16 * 16 grids and each contains 900 pixels)
       
        
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
            initial_states = self.get_initial_states(X) #build an all-zero tensor of shape (samples, output_dim)
        
        
        

        last_output, outputs, states = self.rnn_grid(self.step, inner_data, self.grid_shape,initial_states,
                                             go_backwards=self.go_backwards,
                                             masking=masking)
        
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))
        
        if self.return_sequences:
            return K.reshape(outputs, (-1,) + (np.prod(self.grid_shape), self.input_shape[2]))
        else:
            return last_output

    def rnn_grid(self ,step_function, inputs, grid_shape,initial_states,
        go_backwards=False, masking=True):
        row_states = initial_states[0:grid_shape[0]]
        col_states = initial_states[grid_shape[0]:]
      
        inputs = inputs.dimshuffle((2,1, 0, 3))  # col, row, nsample, dim
        #inputs = input.dimshuffle(tuple(range(1,len(grid_shape)+1))+(0, len(grid_shape)+1) )
                
        def _step(input,row, row_state, col_state): #[row_states] + [col_state]
            #'''it needs the left, and upper grid as it's predecessor,  
			#   it only update the col_state after process one local patch.
			#   although it needs two states to compute each step, but it actually only updates the upper column state, for the
			#   left row_state, it only use it.
			#'''
            states = [col_state]
            output, new_states = step_function(input, row_state, states[0])  # this function actually should only return one state

            if masking:
                # if all-zero input timestep, return
                # all-zero output and unchanged states
                switch = T.any(input, axis=-1, keepdims=True)
                output = T.switch(switch, output, 0. * output)
                return_states = []
                for state, new_state in zip(states, new_states):
                    return_states.append(T.switch(switch, new_state, state))
                return [output] + return_states
            else:
                return [output] + new_states

             
        def loop_over_col(coldata, col_ind,col_state, rows_states, rows_model):
            #'''when loops over a column, it needs the left column as row_states,
			#row_states, col_states are already concatenated to be tensors for scan to process before call this function.
			#it return the results as well as this column response as the row_states to next column.
			#it should return the last column_state
			#'''
            results , _ = theano.scan( fn = _step,
                                       sequences = [coldata, rows_model, rows_states], 									   
                                       outputs_info=[None] +  [col_state] 
                                       )
            new_row_states = results[1:] #'''list of tensor of row_size * nsample * out_dim, but we need to modify it to be list'''
            col_vals = results[0]  
		   #'''tensor of row_size * nsample * out_dim'''
		   #'''the length is the number of states for each single actual step'''
            single_state_num = len(new_row_states) 
            returned_row_states = []
            for timestep_id in range(self.grid_shape[0]):		
			    #'''new_row_states is a list of tensor of size row_size * nsample * out_dim
				#   returned states should be [timestep_1_state_1,...,timestep_1_state_n, ...., timestep_m_state_1,...,timestep_m_state_n],
				#   n is the number of required states for each single actual step function, most time it is 1. but CWRNN has 2.
				#   m is the row_size, = grid_shape[0]
				#'''
			   for state_id in range(single_state_num):
				    returned_row_states.append(T.squeeze(new_row_states[state_id][timestep_id]))
            returned_col_state = []
            for state_id in range(single_state_num):
			  returned_col_state.append(T.squeeze(new_row_states[state_id][-1]))
            #'''output should be list corresponding to the outputs_info'''
            return [col_vals] + [T.stack(returned_row_states, axis =0)] + [T.stack(returned_col_state, axis =0)]		
            
        inputs_model = dict(input = inputs, taps=[0])
        rows_model = T.arange(self.grid_shape[0])
        cols_model = T.arange(self.grid_shape[1])
        #'''loop_over_col(coldata, col_ind, rows_states, col_states, rows_model)'''
		#'''Maybe I need to concatenate all states list to tensor at this stage, scan deal with list of tensor variable, not
		#list of list.'''
        results, _ = theano.scan( fn = loop_over_col,
                                    sequences = [inputs_model, cols_model, T.stack(col_states, axis =0)],
								#'''return grid_results, row_states, col_state'''
                                    outputs_info=[None, T.stack(row_states, axis =0), None],
                                    non_sequences = [rows_model] 
            )
        #update col_states here
		#'''results are list of [grid_tensor, row_state_tensor, col_state]'''
        outputs =  results[0]  #'''tensor type'''
        row_states_tensor  = results[1][-1]  #'''we only need the last column as the row_states, tensor type'''
        col_states_tensor  = results[2]  #'''tensor type'''
        
        
        # deal with Theano API inconsistency
        #if type(results) is list:
        #    outputs = results[0]
        #    states = results[1:]
        #else:
        #    outputs = results
        #    states = []
        outputs = T.squeeze(outputs)
        last_output = outputs[-1,-1]

        #outputs  col, row, nsample, dim
        outputs = outputs.dimshuffle((2,1,0,3))
        
        retuned_row_states = []
        retuned_col_states = []
        
        for state_id in range(len(row_states)):
            state = row_states_tensor[state_id]
            retuned_row_states.append(T.squeeze(state[-1]))
        for state_id in range(len(col_states)):
            state = col_states_tensor[state_id]
            retuned_col_states.append(T.squeeze(state[-1]))

        return last_output, outputs, retuned_row_states + retuned_col_states
        

    def get_config(self):
        config = {"grid_shape": self.grid_shape,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__}
        base_config = super(MultiDimRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

            
