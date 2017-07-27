
import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne
from regularization import *

# copied from Joachim Ott rounding_rnn.py
class Gate(lasagne.layers.Gate):
    """
    This class extends the Lasagne Gate to support rounding of weights
    """
    def __init__(self,mode='normal',H=1.0,nonlinearity=lasagne.nonlinearities.sigmoid,bias_init=lasagne.init.Constant(-1.), **kwargs):
        if mode=='binary':
            if nonlinearity==lasagne.nonlinearities.tanh:
                nonlinearity=binary_tanh_unit
            elif nonlinearity==lasagne.nonlinearities.sigmoid:
                nonlinearity=binary_sigmoid_unit

        super(Gate, self).__init__(nonlinearity=nonlinearity,b=bias_init, **kwargs)



class GRULayer(lasagne.layers.GRULayer):
    """
    This class extends the Lasagne GRULayer to support rounding of weights
    """
    def __init__(self, incoming, num_units,
        stochastic = True, H='glorot',W_LR_scale="Glorot",mode='normal',integer_bits=0,fractional_bits=1,
                 random_seed=666,batch_norm=True,round_hid=True,bn_gamma=lasagne.init.Constant(0.1),mean_substraction_rounding=False,round_bias=True,round_input_weights=True,round_activations=False, **kwargs):

        self.H=H
        self.mode=mode
        self.srng = RandomStreams(random_seed)
        self.stochastic=stochastic
        self.integer_bits=integer_bits
        self.fractional_bits=fractional_bits
        self.batch_norm=batch_norm
        self.round_hid=round_hid
        self.mean_substraction_rounding=mean_substraction_rounding
        self.round_bias=round_bias
        self.round_input_weights=round_input_weights
        self.round_activations=round_activations

        print "Round HID: "+str(self.round_hid)


        if not(mode=='binary' or mode=='ternary' or mode=='dual-copy' or mode=='normal' or mode=='quantize'):
            raise AssertionError("Unexpected value of 'mode'!", mode)

        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            print 'num_inputs: ',num_inputs
            self.W_LR_scale = np.float32(1./np.sqrt(1.5/ (num_inputs + num_units)))

        if mode=='binary' or mode=='ternary' or mode=='dual-copy' or mode=='quantize':
            super(GRULayer, self).__init__(incoming, num_units, **kwargs)
            # add the binary tag to weights
            if self.round_input_weights:
                self.params[self.W_in_to_updategate]=set(['binary'])
                self.params[self.W_in_to_resetgate]=set(['binary'])
                self.params[self.W_in_to_hidden_update]=set(['binary'])
            if self.round_hid:
                self.params[self.W_hid_to_updategate]=set(['binary'])
                self.params[self.W_hid_to_resetgate]=set(['binary'])
                self.params[self.W_hid_to_hidden_update]=set(['binary'])
            if self.round_bias:
                self.params[self.b_updategate] = set(['binary'])
                self.params[self.b_resetgate] = set(['binary'])
                self.params[self.b_hidden_update] = set(['binary'])


        else:
            super(GRULayer, self).__init__(incoming, num_units, **kwargs)

        self.high = np.float32(np.sqrt(6. / (num_inputs + num_units)))
        self.high_hid = np.float32(np.sqrt(6. / (num_units + num_units)))
        self.W0 = np.float32(self.high)
        self.W0_hid = np.float32(self.high_hid)


        input_shape = self.input_shapes[0]
        print 'Input shape: {0}'.format(input_shape)
        #https://github.com/Lasagne/Lasagne/issues/577
        if self.batch_norm:
            print "BatchNorm activated!"
            self.bn =lasagne.layers.BatchNormLayer(input_shape,axes=(0,1),gamma=bn_gamma)  # create BN layer for correct input shape
            self.params.update(self.bn.params)  # make BN params your params
        else:
            print "BatchNorm deactivated!"


        #Create dummy variables
        self.W_hid_to_updategate_d = T.zeros(self.W_hid_to_updategate.shape, self.W_hid_to_updategate.dtype)
        self.W_hid_to_resetgate_d = T.zeros(self.W_hid_to_resetgate.shape, self.W_hid_to_updategate.dtype)
        self.W_hid_to_hidden_update_d = T.zeros(self.W_hid_to_hidden_update.shape, self.W_hid_to_hidden_update.dtype)
        self.W_in_to_updategate_d = T.zeros(self.W_in_to_updategate.shape, self.W_in_to_updategate.dtype)
        self.W_in_to_resetgate_d = T.zeros(self.W_in_to_resetgate.shape, self.W_in_to_resetgate.dtype)
        self.W_in_to_hidden_update_d = T.zeros(self.W_in_to_hidden_update.shape, self.W_in_to_hidden_update.dtype)
        self.b_updategate_d=T.zeros(self.b_updategate.shape, self.b_updategate.dtype)
        self.b_resetgate_d=T.zeros(self.b_resetgate.shape, self.b_resetgate.dtype)
        self.b_hidden_update_d=T.zeros(self.b_hidden_update.shape, self.b_hidden_update.dtype)



    def get_output_for(self, inputs,deterministic=False, **kwargs):
        if not self.stochastic and not deterministic:
            deterministic=True
        print "deterministic mode: ",deterministic
        def apply_regularization(weights,hid=False):
            current_w0 = self.W0
            if hid:
                current_w0 = self.W0_hid

            if self.mean_substraction_rounding:
                return weights
            elif self.mode == "ternary":

                return ternarize_weights(weights, W0=current_w0, deterministic=deterministic,
                                          srng=self.srng)
            elif self.mode == "binary":
                return binarize_weights(weights, 1., self.srng, deterministic=deterministic)
            elif self.mode == "dual-copy":
                return dual_copy_rounding(weights, self.integer_bits, self.fractional_bits)
            elif self.mode == "quantize":
                return quantize_weights(weights, srng=self.srng,deterministic=deterministic)
            else:
                return weights
        if self.round_input_weights:
            self.Wb_in_to_updategate = apply_regularization(self.W_in_to_updategate)
            self.Wb_in_to_resetgate = apply_regularization(self.W_in_to_resetgate)
            self.Wb_in_to_hidden_update = apply_regularization(self.W_in_to_hidden_update)

        if self.round_hid:
            self.Wb_hid_to_updategate = apply_regularization(self.W_hid_to_updategate,hid=True)
            self.Wb_hid_to_resetgate = apply_regularization(self.W_hid_to_resetgate,hid=True)
            self.Wb_hid_to_hidden_update = apply_regularization(self.W_hid_to_hidden_update,hid=True)
        if self.round_bias:
            self.bb_updategate= apply_regularization(self.b_updategate)
            self.bb_resetgate= apply_regularization(self.b_resetgate)
            self.bb_hidden_update=apply_regularization(self.b_hidden_update)


        #Backup high precision values

        Wr_in_to_updategate = self.W_in_to_updategate
        Wr_in_to_resetgate = self.W_in_to_resetgate
        Wr_in_to_hidden_update = self.W_in_to_hidden_update


        if self.round_hid:
            Wr_hid_to_updategate = self.W_hid_to_updategate
            Wr_hid_to_resetgate = self.W_hid_to_resetgate
            Wr_hid_to_hidden_update = self.W_hid_to_hidden_update
        if self.round_bias:
            self.br_updategate = self.b_updategate
            self.br_resetgate = self.b_resetgate
            self.br_hidden_update = self.b_hidden_update

        #Overwrite weights with binarized weights
        if self.round_input_weights:
            self.W_in_to_updategate = self.Wb_in_to_updategate
            self.W_in_to_resetgate = self.Wb_in_to_resetgate
            self.W_in_to_hidden_update = self.Wb_in_to_hidden_update

        if self.round_hid:
            self.W_hid_to_updategate = self.Wb_hid_to_updategate
            self.W_hid_to_resetgate = self.Wb_hid_to_resetgate
            self.W_hid_to_hidden_update = self.Wb_hid_to_hidden_update
        if self.round_bias:
            self.b_updategate = self.bb_updategate
            self.b_resetgate = self.bb_resetgate
            self.b_hidden_update = self.bb_hidden_update

        # Retrieve the layer input
        input = inputs[0]

        #Apply BN
        #https://github.com/Lasagne/Lasagne/issues/577
        if self.batch_norm:
            input = self.bn.get_output_for(input,deterministic=deterministic, **kwargs)
            if len(inputs) > 1:
                new_inputs=[input,inputs[1]]
            else:
                new_inputs=[input]
        else:
            new_inputs=inputs

        inputs=new_inputs

        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape



        # Stack input weight matrices into a (num_inputs, 3*num_units)
        #matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate+self.W_in_to_resetgate_d, self.W_in_to_updategate+self.W_in_to_updategate_d,
             self.W_in_to_hidden_update+self.W_in_to_hidden_update_d], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate+self.W_hid_to_resetgate_d, self.W_hid_to_updategate+self.W_hid_to_updategate_d,
             self.W_hid_to_hidden_update+self.W_hid_to_hidden_update_d], axis=1)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate+self.b_resetgate_d, self.b_updategate+self.b_updategate_d,
             self.b_hidden_update+self.b_hidden_update_d], axis=0)

        if self.precompute_input:

            input = T.dot(input, W_in_stacked) + b_stacked

        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n * self.num_units:(n + 1) * self.num_units]

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(input_n, hid_previous, *args):

            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Reset and update gates
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)

            hidden_update = hidden_update_in + resetgate * hidden_update_hid
            if self.grad_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update)


            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate) * hid_previous + updategate * hidden_update

            return hid

        def step_masked(input_n, mask_n, hid_previous, *args):
            hid = step(input_n, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            hid = T.switch(mask_n, hid, hid_previous)

            return hid


        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = [input]
            step_fun = step

        if not isinstance(self.hid_init, lasagne.layers.Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = lasagne.utils.unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]


        #Copy back high precision values
        if self.round_input_weights:
            self.W_in_to_updategate = Wr_in_to_updategate
            self.W_in_to_resetgate = Wr_in_to_resetgate
            self.W_in_to_hidden_update = Wr_in_to_hidden_update

        if self.round_hid:
            self.W_hid_to_updategate = Wr_hid_to_updategate
            self.W_hid_to_resetgate = Wr_hid_to_resetgate
            self.W_hid_to_hidden_update = Wr_hid_to_hidden_update
        if self.round_bias:
            self.b_updategate = self.br_updategate
            self.b_resetgate = self.br_resetgate
            self.b_hidden_update = self.br_hidden_update

        return hid_out


# extend lasagne.layers.RecurrentLayer and LSTMLayer
class RecurrentLayer(lasagne.layers.RecurrentLayer):
    """
    This class extends the lasagne RecurrentLayer to support rounding of weights
    """
    def __init__(self, incoming, num_units, 
        stochastic=True, H='glorot', W_LR_scale="Glorot",mode='normal',integer_bits=0,fractional_bits=1, 
                random_seed=666,batch_norm=True,round_hid=True,bn_gamma=lasagne.init.Constant(0.1),mean_substraction_rounding=False,round_bias=True,round_input_weights=True,round_activations=False, **kwargs):

        self.H=H
        self.mode=mode
        self.srng=RandomStreams(random_seed)
        self.stochastic=stochastic
        self.integer_bits=integer_bits
        self.fractional_bits=fractional_bits
        self.batch_norm=batch_norm
        self.round_hid=round_hid
        self.mean_substraction_rounding=mean_substraction_rounding
        self.round_bias=round_bias
        self.round_input_weights=round_input_weights
        self.round_activations=round_activations

        print "Round HID: "+str(self.round_hid)

        if not(mode=='bianry' or mode=='ternary' or mode=='dual-copy' or mode=='normal' or mode == 'quantize'):
            raise AssertionError("Unexpected value of 'mode' ! ", mode)

        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            print 'num_inputs: ',num_inputs
            self.W_LR_scale = np.float32(1./np.sqrt(1.5/ (num_inputs + num_units)))

        if mode=='bianry' or mode=='ternary' or mode=='dual-copy' or mode=='quantize' :
            super(RecurrentLayer, self).__init__(incoming, num_units, **kwargs)
            # add the bianry tag to weights
            if self.round_input_weights:
                self.params[self.W_in_to_hid]=set(['binary'])
            if self.round_hid:
                self.params[self.W_hid_to_hid]=set(['binary'])
            if self.round_bias:
                self.params[self.b]=set(['binary'])

        else :
            super(RecurrentLayer, self).__init__(incoming, num_units, **kwargs)

        self.high = np.float32(np.sqrt(6. / (num_inputs + num_units)))
        self.high_hid = np.float32(np.sqrt(6. / (num_inputs + num_units)))
        self.w0 = np.float32(self.high)
        self.w0_hid = np.float32(self.high_hid)

        input_shape = self.input_shapes[0]
        print 'Input shape: {0}'.format(input_shape)
        #http://github.com/Lasagne/Lasagne/issues/577
        if self.batch_norm:
            print "BatchNorm activated!"
            self.bn = lasagne.layers.BatchNormLayer(input_shape,axes=(0,1),gamma=bn_gamma)
            self.params.update(self.bn.params)
        else:
            print "BatchNorm deactivated!"

        self.W_in_to_hid_d=T.zeros(self.W_in_to_hid.shape, self.W_in_to_hid.dtype)
        self.W_hid_to_hid_d=T.zeros(self.W_hid_to_hid.shape, self.W_hid_to_hid.dtype)
        self.b_d=T.zeros(self.b.shape, self.b.dtype)


    def get_output_for(self, inputs,deterministic=False, **kwargs):
        if not self.stochastic and not deterministic:
            deterministic=True
        print "deterministic mode: ", deterministic
        def apply_regularization(weights,hid=False):
            current_w0 = self.w0
            if hid:
                current_w0=self.w0_hid

            if self.mean_substraction_rounding:
                return weights
            elif self.mode == 'ternary':

                return ternarize_weights(weights, w0=current_w0, deterministic=deterministic,srng=self.srng)

            elif self.mode == "binary":
                return binarize_weights(weights, 1., self.srng, deterministic=deterministic)
            elif self.mode == "dual-copy":
                return quantize_weights(weights, srng=self.srng, deterministic=deterministic)
            else:
                return weights
        if self.round_input_weights:
            self.Wb_in_to_hid = apply_regularization(self.W_in_to_hid)

        if self.round_hid:
            self.Wb_hid_to_hid = apply_regularization(self.W_hid_to_hid)

        if self.round_bias:
            self.bb = apply_regularization(self.b)



        Wr_in_to_hid = self.W_in_to_hid

        if self.round_hid:
            Wr_hid_to_hid = self.W_hid_to_hid
        if self.round_bias:
            self.br = self.b

        if self.round_input_weights:
            self.W_in_to_hid = self.Wb_in_to_hid

        if self.round_hid:
            self.W_hid_to_hid = self.Wb_hid_to_hid

        if self.round_bias:
            self.b = self.bb


        input = inputs[0]


        if self.batch_norm:
            input = self.bn.get_output_for(input,deterministic=deterministic, **kwargs)
            if len(inputs) > 1:
                new_inputs=[input,inputs[1]]
            else:
                new_inputs=[input]
        else:
            new_inputs=inputs

        inputs=new_inputs


        input = inputs[0]

        mask = None
        hid_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        if input.ndim > 3:
            input = T.flatten(input, 3)


        input = input.dimshuffle(1,0,2)
        seq_len , num_batch, _ = input.shape


        W_in_stacked = T.concatenate(
            [self.W_in_to_hid + self.W_in_to_hid_d], axis=1)

        W_hid_stacked = T.concatenate(
            [self.W_hid_to_hid + self.W_hid_to_hid_d], axis=1)

        b_stacked = T.concatenate(
            [self.b + self.b_d], axis=0)

        if self.precompute_input:

            input = T.dot(input, W_in_stacked) + b_stacked



        def step(input_n, hid_previous, *args):

            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping,self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:

                input_n = T.dot(input_n, W_in_stacked) + b_stacked


            hid = self.nonlinearity(hid_input + input_n)

            return hid

        def step_masked(input_n, mask_n, hid_previous, *args):
            hid = step(input_n, hid_previous, *args)

            hid = T.switch(mask_n, hid, hid_previous)

            return hid


        if mask is not None:


            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked

        else:
            sequences = [input]
            step_fun = step

        if not isinstance(self.hid_init, lasagne.layers.Layer):

            hid_init = T.dot(T.ones((num_batch,1)), self.hid_init)

        non_seqs = [W_hid_stacked]


        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:

            input_shape = self.input_shapes[0]

            hid_out = lasagne.utils.unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:


            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:

            hid_out = hid_out.dimshuffle(1, 0, 2)

            if self.backwards:
                hid_out = hid_out[:, ::-1]


        if self.round_input_weights:
            self.W_in_to_hid = Wr_in_to_hid

        if self.round_hid:
            self.W_hid_to_hid = Wr_hid_to_hid

        if self.round_bias:
            self.b = self.br

        return hid_out


class LSTMLayer(lasagne.layers.LSTMLayer):
    """
    This class extends the lasagne LSTMLayer to support rounding of weights
    """
    def __init__(self, incoming, num_units, 
        stochastic = True, H='glorot',W_LR_scale="Glorot",mode='normal',integer_bits=0,fractional_bits=1,
                random_seed=666,batch_norm=True,round_hid=True,bn_gamma=lasagne.init.Constant(0.1),mean_substraction_rounding=False,round_bias=True,round_input_weights=True,round_activations=False, **kwargs):

        self.H=H
        self.mode=mode
        self.srng = RandomStreams(random_seed)
        self.stochastic=stochastic
        self.integer_bits=integer_bits
        self.fractional_bits=fractional_bits
        self.batch_norm=batch_norm
        self.round_hid=round_hid
        self.mean_substraction_rounding=mean_substraction_rounding
        self.round_bias=round_bias
        self.round_input_weights=round_input_weights
        self.round_activations=round_activations
        

        print "Round HID: "+str(self.round_hid)

        if not(mode=='binary' or mode=='ternary' or mode=='dual-copy' or mode=='normal' or mode=='quantize'):
            raise AssertionError("Unexpected value of 'mode' ! ", mode)

        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            print 'num_inputs: ',num_inputs
            self.W_LR_scale = np.float32(1./np.sqrt(1.5/ (num_inputs + num_units)))

        if mode=='binary' or mode=='ternary' or mode=='dual-copy' or mode=='quantize':
            super(LSTMLayer, self).__init__(incoming, num_units, **kwargs)
            #add the binary tag to weights
            if self.round_input_weights:
                self.params[self.W_in_to_ingate]=set(['binary'])
                self.params[self.W_in_to_forgetgate]=set(['binary'])
                self.params[self.W_in_to_cell]=set(['binary'])
                self.params[self.W_in_to_outgate]=set(['binary'])
            if self.round_hid:
                self.params[self.W_hid_to_ingate]=set(['binary'])
                self.params[self.W_hid_to_forgetgate]=set(['binary'])
                self.params[self.W_hid_to_cell]=set(['binary'])
                self.params[self.W_hid_to_outgate]=set(['binary'])
            if self.round_bias:
                self.params[self.b_ingate] = set(['binary'])
                self.params[self.b_forgetgate] = set(['binary'])
                self.params[self.b_cell] = set(['binary'])
                self.params[self.b_outgate] = set(['binary'])

        else:
            super(LSTMLayer, self).__init__(incoming, num_units, **kwargs)

        self.high = np.float32(np.sqrt(6. / (num_inputs + num_units)))
        self.high_hid = np.float32(np.sqrt(6. / (num_units + num_units)))
        self.W0 = np.float32(self.high)
        self.W0_hid = np.float32(self.high_hid)


        input_shape = self.input_shapes[0]
        print 'Input Shape: {0}'.format(input_shape)
        #https://github.com/Lasagne/Lasagne/issues/577
        if self.batch_norm:
            print "BatchNorm activated!"
            self.bn = lasagne.layers.BatchNormLayer(input_shape, axes=(0,1),gamma=bn_gamma)
            self.params.update(self.bn.params)
        else:
            print "BatchNorm deactivated!"


        self.W_hid_to_ingate_d = T.zeros(self.W_hid_to_ingate.shape, self.W_hid_to_ingate.dtype)
        self.W_hid_to_forgetgate_d = T.zeros(self.W_hid_to_forgetgate.shape, self.W_hid_to_forgetgate.dtype)
        self.W_hid_to_cell_d = T.zeros(self.W_hid_to_cell.shape, self.W_hid_to_cell.dtype)
        self.W_hid_to_outgate_d = T.zeros(self.W_hid_to_outgate.shape, self.W_hid_to_outgate.dtype)
        self.W_in_to_ingate_d = T.zeros(self.W_in_to_ingate.shape, self.W_in_to_ingate.dtype)
        self.W_in_to_forgetgate_d = T.zeros(self.W_in_to_forgetgate.shape, self.W_in_to_forgetgate.dtype)
        self.W_in_to_cell_d = T.zeros(self.W_in_to_cell.shape, self.W_in_to_cell.dtype)
        self.W_in_to_outgate_d = T.zeros(self.W_in_to_outgate.shape, self.W_in_to_outgate.dtype)
        self.b_ingate_d = T.zeros(self.b_ingate.shape, self.b_ingate.dtype)
        self.b_forgetgate_d = T.zeros(self.b_forgetgate.shape, self.b_forgetgate.dtype)
        self.b_cell_d = T.zeros(self.b_cell.shape, self.b_cell.dtype)
        self.b_outgate_d = T.zeros(self.b_outgate.shape, self.b_outgate.dtype)


    def get_output_for(self, inputs, deterministic=False, **kwargs):
        if not self.stochastic and not deterministic:
            deterministic=True
        print "deterministic mode: ",deterministic
        def apply_regularization(weights,hid=False):
            current_W0 = self.W0
            if hid:
                current_W0 = self.W0_hid

            if self.mean_substraction_rounding:
                return weights
            elif self.mode == "ternary":

                return ternarize_weights(weights, W0=current_W0, deterministic=deterministic,
                                        srng=self.srng)
            elif self.mode == "binary":
                return binarize_weights(weights, 1., self.srng, deterministic=deterministic)
            elif self.mode == "dual-copy":
                return dual_copy_rounding(weights, self.integer_bits, self.fractional_bits)
            elif self.mode == "quantize":
                return quantize_weights(weights, srng=self.srng, deterministic=deterministic)
            else:
                return weights

        if self.round_input_weights:
            self.Wb_in_to_ingate = apply_regularization(self.W_in_to_ingate)
            self.Wb_in_to_forgetgate = apply_regularization(self.W_in_to_forgetgate)
            self.Wb_in_to_cell = apply_regularization(self.W_in_to_cell)
            self.Wb_in_to_outgate = apply_regularization(self.W_in_to_outgate)

        if self.round_hid:
            self.Wb_hid_to_ingate = apply_regularization(self.W_hid_to_ingate)
            self.Wb_hid_to_forgetgate = apply_regularization(self.W_hid_to_forgetgate)
            self.Wb_hid_to_cell = apply_regularization(self.W_hid_to_cell)
            self.Wb_hid_to_outgate = apply_regularization(self.W_hid_to_outgate)

        if self.round_bias:
            self.bb_ingate = apply_regularization(self.b_ingate)
            self.bb_forgetgate = apply_regularization(self.b_forgetgate)
            self.bb_cell = apply_regularization(self.b_cell)
            self.bb_outgate = apply_regularization(self.b_outgate)


        #Backup high precision values
        Wr_in_to_ingate = self.W_in_to_ingate
        Wr_in_to_forgetgate = self.W_in_to_forgetgate
        Wr_in_to_cell = self.W_in_to_cell
        Wr_in_to_outgate = self.W_in_to_outgate

        if self.round_hid:
            Wr_hid_to_ingate = self.W_hid_to_ingate
            Wr_hid_to_forgetgate = self.W_hid_to_forgetgate
            Wr_hid_to_cell = self.W_hid_to_cell
            Wr_hid_to_outgate = self.W_hid_to_outgate

        if self.round_bias:
            self.br_ingate = self.b_ingate
            self.br_forgetgate = self.b_forgetgate
            self.br_cell = self.b_cell
            self.br_outgate = self.b_outgate

        #Overwrite weights with binarized weights
        if self.round_input_weights:
            self.W_in_to_ingate = self.Wb_in_to_ingate
            self.W_in_to_forgetgate = self.Wb_in_to_forgetgate
            self.W_in_to_cell = self.Wb_in_to_cell
            self.W_in_to_outgate = self.Wb_in_to_outgate

        if self.round_hid:
            self.W_hid_to_ingate = self.Wb_hid_to_ingate
            self.W_hid_to_forgetgate = self.Wb_hid_to_forgetgate
            self.W_hid_to_cell = self.Wb_hid_to_cell
            self.W_hid_to_outgate = self.Wb_hid_to_outgate

        if self.round_bias:
            self.b_ingate = self.bb_ingate
            self.b_forgetgate = self.bb_forgetgate
            self.b_cell = self.bb_cell
            self.b_outgate = self.bb_outgate


        #Retrieve the layer input
        input = inputs[0]

        #Apply BN
        #https://github.com/Lasagne/Lasagne/issues/577
        if self.batch_norm:
            input = self.bn.get_output_for(input,deterministic=deterministic, **kwargs)
            if len(inputs) > 1:
                new_inputs=[input,inputs[1]]
            else:
                new_inputs=[input]
        else:
            new_inputs=inputs

        inputs=new_inputs

        #Retrieve the layer input
        input = inputs[0]
        #Retrieve the mask when it is supplied
        mask = None 
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        #Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to 
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 4 * num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate+self.W_in_to_ingate_d, self.W_in_to_forgetgate+self.W_in_to_forgetgate_d,
            self.W_in_to_cell+self.W_in_to_cell_d, self.W_in_to_outgate+self.W_in_to_outgate_d], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate+self.W_hid_to_ingate_d, self.W_hid_to_forgetgate+self.W_hid_to_forgetgate_d,
            self.W_hid_to_cell+self.W_hid_to_cell_d, self.W_hid_to_outgate+self.W_hid_to_outgate_d], axis=1)

        # Stack gate biases into a (4 * num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate+self.b_ingate_d, self.b_forgetgate+self.b_forgetgate_d,
            self.b_cell+self.b_cell_d, self.b_outgate+self.b_outgate_d], axis=0)

        if self.precompute_input:

            input = T.dot(input, W_in_stacked) + b_stacked

        # at each call to scan , input_n will be (n_time-steps, 4*num_units).
        # we define a slicing funcitno that extract the input to each LSTM gate
        def slice_w(x,n):
            s = x[:, n * self.num_units:(n + 1) * self.num_units]
            if self.num_units == 1:
                s = T.addbroadcast(s, 1)
            return s

        
        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, hid_previous, cell_previous, *arg):
            # compute W_{hi} h_{t-1}, W_{hf} h_{t-1}, W_{hc} h_{t-1}, W_{ho} h_{t-1}
            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                # Compute W_{xi}x_t + b_{i}, W_{xf}x_t + b_{f}, W_{xc}x_t + b_{c}, W_{xo}x_t + b{o}
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # input, forget and output gates
            ingate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            forgetgate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            outgate = slice_w(hid_input, 3) + slice_w(input_n, 3)
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            outgate = self.nonlinearity_outgate(outgate)

            # compute new cell state
            cell_input = slice_w(hid_input, 2) + slice_w(input_n, 2)
            cell_input = self.nonlinearity_cell(cell_input)
            cell = forgetgate * cell_previous + ingate * cell_input


            # compute o_t emul nonlinearity(c_t)
            hid = outgate * self.nonlinearity(cell)

            return [cell, hid]

        def step_masked(input_n, mask_n, cell_previous, hid_previous, *args):
            cell, hid = step(input_n, cell_previous, hid_previous, *args)

            # skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [cell, hid]

        if mask is not None:



            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = [input]
            step_fun = step

        if not isinstance(self.hid_init, lasagne.layers.Layer):

            hid_init = T.dot(T.ones((num_batch,1)), self.hid_init)

        if not isinstance(self.cell_init, lasagne.layers.Layer):

            cell_init = T.dot(T.ones((num_batch,1)), self.cell_init)

        non_seqs = [W_hid_stacked]

        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:

            input_shape = self.input_shapes[0]

            cell_out, hid_out = lasagne.utils.unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])

        else: 
            cell_out, hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[cell_init, hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        if self.only_return_final:
            hid_out = hid_out[-1]
        else:

            hid_out = hid_out.dimshuffle(1,0,2)

            if self.backwards:
                hid_out = hid_out[:, ::-1]

        #copy back high precision values
        if self.round_input_weights:
            self.W_in_to_ingate = Wr_in_to_ingate
            self.W_in_to_forgetgate = Wr_in_to_forgetgate
            self.W_in_to_cell = Wr_in_to_cell
            self.W_in_to_outgate = Wr_in_to_outgate

        if self.round_hid:
            self.W_hid_to_ingate = Wr_hid_to_ingate
            self.W_hid_to_forgetgate = Wr_hid_to_forgetgate
            self.W_hid_to_cell = Wr_hid_to_cell
            self.W_hid_to_outgate = Wr_hid_to_outgate

        if self.round_bias:
            self.b_ingate = self.br_ingate
            self.b_forgetgate = self.br_forgetgate
            self.b_cell = self.br_cell
            self.b_outgate = self.br_outgate

        return hid_out



# copied from Joachim Ott rounding_rnn.py
# This function computes the gradient of the binary weights
def compute_rnn_grads(loss,network):

    layers = lasagne.layers.get_all_layers(network)
    grads = []

    for layer in layers:

        params = layer.get_params(binary=True)
        if params:
            for param in params:
                print(param.name)
                if param.name=='W_in_to_updategate':
                    grads.append(theano.grad(loss, wrt=layer.W_in_to_updategate_d))
                elif param.name=='W_in_to_resetgate':
                    grads.append(theano.grad(loss, wrt=layer.W_in_to_resetgate_d))
                elif param.name=='W_in_to_hidden_update':
                    grads.append(theano.grad(loss, wrt=layer.W_in_to_hidden_update_d))
                elif param.name=='W_hid_to_updategate':
                    grads.append(theano.grad(loss, wrt=layer.W_hid_to_updategate_d))
                elif param.name=='W_hid_to_resetgate':
                    grads.append(theano.grad(loss, wrt=layer.W_hid_to_resetgate_d))
                elif param.name=='W_hid_to_hidden_update':
                    grads.append(theano.grad(loss, wrt=layer.W_hid_to_hidden_update_d))
                elif param.name=='b_updategate':
                    grads.append(theano.grad(loss, wrt=layer.b_updategate_d))
                elif param.name == 'b_resetgate':
                    grads.append(theano.grad(loss, wrt=layer.b_resetgate_d))
                elif param.name == 'b_hidden_update':
                    grads.append(theano.grad(loss, wrt=layer.b_hidden_update_d))

    return grads


#Copied from BinaryNet by Matthieu Courbariaux
def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)

# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1
# during back propagation
def binary_tanh_unit(x):
    return 2.*GradPreserveRoundTensor(hard_sigmoid(x))-1.

def binary_sigmoid_unit(x):
    return GradPreserveRoundTensor(hard_sigmoid(x))