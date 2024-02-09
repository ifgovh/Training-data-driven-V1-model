import numpy as np
import tensorflow as tf


def gauss_pseudo(v_scaled, sigma, amplitude):
    return tf.math.exp(-tf.square(v_scaled) / tf.square(sigma)) * amplitude


def pseudo_derivative(v_scaled, dampening_factor):
    return dampening_factor * tf.maximum(1 - tf.abs(v_scaled), 0)


@tf.custom_gradient
def spike_gauss(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)
        # dz_dv_scaled = pseudo_derivative(v_scaled, .3)
        # dz_dv_scaled = 1.

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled,
                tf.zeros_like(sigma), tf.zeros_like(amplitude)]

    return tf.identity(z_, name='spike_gauss'), grad


def exp_convolve(tensor, decay=.8, reverse=False, initializer=None, axis=0):
    rank = len(tensor.get_shape())
    perm = np.arange(rank)
    perm[0], perm[axis] = perm[axis], perm[0]
    tensor = tf.transpose(tensor, perm)

    if initializer is None:
        initializer = tf.zeros_like(tensor[0])

    def scan_fun(_acc, _t):
        return _acc * decay + _t

    filtered = tf.scan(scan_fun, tensor, reverse=reverse, initializer=initializer)

    filtered = tf.transpose(filtered, perm)
    return filtered


class SparseLayer(tf.keras.layers.Layer):
    def __init__(self, indices, weights, dense_shape, bkg_weights, down_sampled_decode_noise_path=None, use_decoded_noise=False, dtype=tf.float32, scale=[1,1], **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self._indices = indices
        self._weights = weights
        self._dense_shape = dense_shape
        self._max_batch = int(2**31 / weights.shape[0])
        self._dtype = dtype
        self._bkg_weights = bkg_weights
        self._use_decoded_noise = use_decoded_noise
        if use_decoded_noise:
            from scipy.io import loadmat
            tmp = loadmat(down_sampled_decode_noise_path)
            self.noise_data = tf.convert_to_tensor(tmp['additive_noise'].reshape(-1), dtype=self._compute_dtype)

    def call(self, inp):
        tf_shp = tf.unstack(tf.shape(inp))
        shp = inp.shape.as_list()
        for i, a in enumerate(shp):
            if a is None:
                shp[i] = tf_shp[i]

        sparse_w_in = tf.sparse.SparseTensor(
            self._indices, self._weights, self._dense_shape)
        inp = tf.reshape(inp, (shp[0] * shp[1], shp[2]))

        input_current = tf.sparse.sparse_dense_matmul(sparse_w_in, tf.cast(inp, tf.float32), adjoint_b=True)
        input_current = tf.transpose(input_current)
        input_current = tf.cast(input_current, self._dtype)
        if self._use_decoded_noise:
            # quick noise: sample every step
            gen_ind_quick = tf.random.uniform(shape=(shp[0], shp[1], self._dense_shape[0]), maxval=28406000, dtype=tf.int64) # batch, seq_len, neurons*4
            # slow noise: sample every trial
            gen_ind_slow = tf.random.uniform(shape=(shp[0], 1, self._dense_shape[0]), maxval=28406000, dtype=tf.int64) # batch, 1, neurons*4
            gen_ind_slow = tf.tile(gen_ind_slow,[1,shp[1],1]) # batch, seq_len, neurons*4
            quick_noise = tf.gather(self.noise_data, gen_ind_quick)
            slow_noise = tf.gather(self.noise_data, gen_ind_slow)

            noise_input = tf.cast(tf.ones_like(self._bkg_weights[None, None])*self.scale[0], self._compute_dtype) * quick_noise + \
                tf.cast(tf.ones_like(self._bkg_weights[None, None])*self.scale[1], self._compute_dtype) * slow_noise
        else:
            rest_of_brain = tf.reduce_sum(tf.cast(
                tf.random.uniform((shp[0], shp[1], 10)) < .1, self._compute_dtype), -1)
            noise_input = tf.cast(
                self._bkg_weights[None, None], self._compute_dtype) * rest_of_brain[..., None] / 10.

        input_current = tf.reshape(
            input_current, (shp[0], shp[1], -1)) + noise_input
        return input_current

class SignedConstraint(tf.keras.constraints.Constraint):
    def __init__(self, positive):
        self._positive = positive

    def __call__(self, w):
        sign_corrected_w = tf.where(self._positive, tf.nn.relu(w), -tf.nn.relu(-w))
        return sign_corrected_w


class SparseSignedConstraint(tf.keras.constraints.Constraint):
    def __init__(self, mask, positive):
        self._mask = mask
        self._positive = positive

    def __call__(self, w):
        sign_corrected_w = tf.where(self._positive, tf.nn.relu(w), -tf.nn.relu(-w))
        return tf.where(self._mask, sign_corrected_w, tf.zeros_like(sign_corrected_w))


class StiffRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, strength, initial_value):
        super().__init__()
        self._strength = strength
        self._initial_value = tf.Variable(initial_value, trainable=False)

    def __call__(self, x):
        return self._strength * tf.reduce_sum(tf.square(x - self._initial_value))

class L2Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, strength):
        super().__init__()
        self._strength = strength

    def __call__(self, x):
        return self._strength * tf.nn.l2_loss(x)

class BillehColumn(tf.keras.layers.Layer):
    def __init__(self, network, input_population, bkg_weights,
                 dt=1., gauss_std=.5, dampening_factor=.3,
                 input_weight_scale=1., recurrent_weight_scale=1.,
                 spike_gradient=False, max_delay=5, train_recurrent=True, train_input=True, 
                 train_bkg=False, use_dale_law=True, _return_interal_variables=False):
        super().__init__()
        self._params = network['node_params']

        voltage_scale = self._params['V_th'] - self._params['E_L']
        voltage_offset = self._params['E_L']
        self._params['V_th'] = (self._params['V_th'] - voltage_offset) / voltage_scale
        self._params['E_L'] = (self._params['E_L'] - voltage_offset) / voltage_scale
        self._params['V_reset'] = (self._params['V_reset'] - voltage_offset) / voltage_scale
        self._params['asc_amps'] = self._params['asc_amps'] / voltage_scale[..., None]

        self._node_type_ids = network['node_type_ids']
        self._dt = dt   

        self._return_interal_variables = _return_interal_variables

        # for random spike, the instantaneous firing rate when v = v_th
        self._spike_gradient = spike_gradient

        n_receptors = network['node_params']['tau_syn'].shape[1]
        self._n_receptors = n_receptors
        self._n_neurons = network['n_nodes']
        self._dampening_factor = tf.cast(dampening_factor, self._compute_dtype)
        self._gauss_std = tf.cast(gauss_std, self._compute_dtype)

        tau = self._params['C_m'] / self._params['g']
        self._decay = np.exp(-dt / tau)
        self._current_factor = 1 / self._params['C_m'] * (1 - self._decay) * tau
        self._syn_decay = np.exp(-dt / np.array(self._params['tau_syn']))
        self._psc_initial = np.e / np.array(self._params['tau_syn'])

        # synapses: target_ids, source_ids, weights, delays

        self.max_delay = int(np.round(np.min([np.max(network['synapses']['delays']), max_delay])))

        self.state_size = (
            self._n_neurons * self.max_delay,  # z buffer
            self._n_neurons,                   # v
            self._n_neurons,                   # r
            self._n_neurons,                   # asc 1
            self._n_neurons,                   # asc 2
            n_receptors * self._n_neurons,     # psc rise
            n_receptors * self._n_neurons,     # psc
        )
        # useless now; it was for training the neuron parameters
        def _f(_v, trainable=False):
            return tf.Variable(tf.cast(self._gather(_v), self._compute_dtype), trainable=trainable)

        def inv_sigmoid(_x):
            return tf.math.log(_x / (1 - _x))

        # useless
        def custom_val(_v, trainable=False):
            _v = tf.Variable(tf.cast(inv_sigmoid(self._gather(_v)), self._compute_dtype), trainable=trainable)

            def _g():
                return tf.nn.sigmoid(_v.read_value())

            return _v, _g

        self.v_reset = _f(self._params['V_reset'])
        self.syn_decay = _f(self._syn_decay)
        self.psc_initial = _f(self._psc_initial)
        self.t_ref = _f(self._params['t_ref'])
        self.asc_amps = _f(self._params['asc_amps'], trainable=False)
        # self.param_k = _f(self._params['k'], trainable=True)
        _k = self._params['k']
        # _k[_k < .0031] = .0007
        self.param_k, self.param_k_read = custom_val(_k, trainable=False)
        self.v_th = _f(self._params['V_th'])
        self.e_l = _f(self._params['E_L'])
        self.param_g = _f(self._params['g'])
        self.decay = _f(self._decay)
        self.current_factor = _f(self._current_factor)
        self.voltage_scale = _f(voltage_scale)
        self.voltage_offset = _f(voltage_offset)

        self.recurrent_weights = None
        self.disconnect_mask = None

        indices, weights, dense_shape = \
            network['synapses']['indices'], network['synapses']['weights'], network['synapses']['dense_shape']
        weights = weights / voltage_scale[self._node_type_ids[indices[:, 0] // self._n_receptors]]
        delays = np.round(np.clip(network['synapses']['delays'], dt, self.max_delay) / dt).astype(np.int32)
        dense_shape = dense_shape[0], self.max_delay * dense_shape[1]
        indices[:, 1] = indices[:, 1] + self._n_neurons * (delays - 1)
        weights = weights.astype(np.float32)
        print(f'> Recurrent synapses {len(indices)}')
        input_weights = input_population['weights'].astype(np.float32)
        input_indices = input_population['indices']
        input_weights = input_weights / voltage_scale[self._node_type_ids[input_indices[:, 0] // self._n_receptors]]
        print(f'> Input synapses {len(input_indices)}')
        input_dense_shape = (self._n_receptors * self._n_neurons, input_population['n_inputs'])

        self.recurrent_weight_positive = tf.Variable(
            weights >= 0., name='recurrent_weights_sign', trainable=False)
        self.input_weight_positive = tf.Variable(
            input_weights >= 0., name='input_weights_sign', trainable=False)
        if use_dale_law:
            self.recurrent_weight_values = tf.Variable(
                weights * recurrent_weight_scale, name='sparse_recurrent_weights',
                constraint=SignedConstraint(self.recurrent_weight_positive),
                trainable=train_recurrent)
        else:
            self.recurrent_weight_values = tf.Variable(
                weights * recurrent_weight_scale, name='sparse_recurrent_weights',
                constraint=None,
                trainable=train_recurrent)
        self.recurrent_indices = tf.Variable(indices, trainable=False)
        self.recurrent_dense_shape = dense_shape

        if use_dale_law:
            self.input_weight_values = tf.Variable(
                input_weights * input_weight_scale, name='sparse_input_weights',
                constraint=SignedConstraint(self.input_weight_positive),
                trainable=train_input)
        else:
            self.input_weight_values = tf.Variable(
                input_weights * input_weight_scale, name='sparse_input_weights',
                constraint=None,
                trainable=train_input)

        self.input_indices = tf.Variable(input_indices, trainable=False)
        self.input_dense_shape = input_dense_shape
        bkg_weights = bkg_weights / np.repeat(voltage_scale[self._node_type_ids], self._n_receptors)
        # this actutually is not used; we used the decoded noise
        self.bkg_weights = tf.Variable(bkg_weights * 10., name='rest_of_brain_weights', trainable=train_bkg)

    def zero_state(self, batch_size, dtype=tf.float32):
        z0_buf = tf.zeros((batch_size, self._n_neurons * self.max_delay), dtype)
        v0 = tf.ones((batch_size, self._n_neurons), dtype) * tf.cast(self.v_th * .0 + 1. * self.v_reset, dtype)
        r0 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_10 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_20 = tf.zeros((batch_size, self._n_neurons), dtype)
        psc_rise0 = tf.zeros((batch_size, self._n_neurons * self._n_receptors), dtype)
        psc0 = tf.zeros((batch_size, self._n_neurons * self._n_receptors), dtype)
        return z0_buf, v0, r0, asc_10, asc_20, psc_rise0, psc0

    def random_state(self, batch_size, dtype=tf.float32):
        z0_buf = tf.cast(tf.random.uniform((batch_size, self._n_neurons * self.max_delay), 0, 2, tf.int32), dtype)
        v0 = tf.random.uniform((batch_size, self._n_neurons), tf.cast(self.v_reset,dtype), tf.cast(self.v_th,dtype), dtype)
        r0 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_10 = tf.random.normal((batch_size, self._n_neurons), mean=-0.28, stddev=1.75, dtype=dtype) # min -87 max 59
        asc_20 = tf.random.normal((batch_size, self._n_neurons), mean=-0.28, stddev=1.75, dtype=dtype)
        psc_rise0 = tf.random.normal((batch_size, self._n_neurons * self._n_receptors), mean=0.29, stddev=0.77, dtype=dtype) #-3.8~33.6
        psc0 = tf.random.normal((batch_size, self._n_neurons * self._n_receptors), mean=1.17, stddev=3.19, dtype=dtype) # -21~147
        return z0_buf, v0, r0, asc_10, asc_20, psc_rise0, psc0

    def _gather(self, prop):
        return tf.gather(prop, self._node_type_ids)

    def call(self, inputs, state, constants=None):
        batch_size = inputs.shape[0]
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
           
        z_buf, v, r, asc_1, asc_2, psc_rise, psc = state

        shaped_z_buf = tf.reshape(z_buf, (-1, self.max_delay, self._n_neurons))
        prev_z = shaped_z_buf[:, 0]

        psc_rise = tf.reshape(psc_rise, (batch_size, self._n_neurons, self._n_receptors))
        psc = tf.reshape(psc, (batch_size, self._n_neurons, self._n_receptors))

        sparse_w_rec = tf.sparse.SparseTensor(
            self.recurrent_indices, self.recurrent_weight_values, self.recurrent_dense_shape)

        i_rec = tf.sparse.sparse_dense_matmul(sparse_w_rec, tf.cast(z_buf, tf.float32), adjoint_b=True)
        i_rec = tf.transpose(i_rec)

        rec_inputs = tf.cast(i_rec, self._compute_dtype)
        rec_inputs = tf.reshape(rec_inputs + inputs, (batch_size, self._n_neurons, self._n_receptors))        

        new_psc_rise = self.syn_decay * psc_rise + rec_inputs * self.psc_initial
        new_psc = psc * self.syn_decay + self._dt * self.syn_decay * psc_rise

        new_r = tf.nn.relu(r + prev_z * self.t_ref - self._dt)

        k = self.param_k_read()
        asc_amps = self.asc_amps
        new_asc_1 = tf.exp(-self._dt * k[:, 0]) * asc_1 + prev_z * asc_amps[:, 0]
        new_asc_2 = tf.exp(-self._dt * k[:, 1]) * asc_2 + prev_z * asc_amps[:, 1]

        reset_current = prev_z * (self.v_reset - self.v_th)
        input_current = tf.reduce_sum(psc, -1)
        decayed_v = self.decay * v

        gathered_g = self.param_g * self.e_l
        c1 = input_current + asc_1 + asc_2 + gathered_g
        new_v = decayed_v + self.current_factor * c1 + reset_current

        normalizer = self.v_th - self.e_l
        v_sc = (new_v - self.v_th) / normalizer
        
        new_z = spike_gauss(v_sc, self._gauss_std, self._dampening_factor)        

        new_z = tf.where(new_r > 0., tf.zeros_like(new_z), new_z)

        new_psc = tf.reshape(new_psc, (batch_size, self._n_neurons * self._n_receptors))
        new_psc_rise = tf.reshape(new_psc_rise, (batch_size, self._n_neurons * self._n_receptors))

        new_shaped_z_buf = tf.concat((new_z[:, None], shaped_z_buf[:, :-1]), 1)
        new_z_buf = tf.reshape(new_shaped_z_buf, (-1, self._n_neurons * self.max_delay))

        if self._return_interal_variables:
            new_ascs = tf.concat((new_asc_1, new_asc_2), -1)
            outputs = (new_z, new_v * self.voltage_scale + self.voltage_offset, new_ascs, new_psc_rise, new_psc)
        else:
            outputs = (new_z, new_v * self.voltage_scale + self.voltage_offset)
        new_state = (new_z_buf, new_v, new_r, new_asc_1, new_asc_2, new_psc_rise, new_psc)

        return outputs, new_state

def huber_quantile_loss(u, tau, kappa):
    branch_1 = tf.abs(tau - tf.cast(u <= 0, tf.float32)) / (2 * kappa) * tf.square(u)
    branch_2 = tf.abs(tau - tf.cast(u <= 0, tf.float32)) * (tf.abs(u) - .5 * kappa)
    return tf.where(tf.abs(u) <= kappa, branch_1, branch_2)


def compute_spike_rate_distribution_loss(_spikes, target_rate):
    _rate = tf.reduce_mean(_spikes, (0, 1))
    ind = tf.range(target_rate.shape[0])
    rand_ind = tf.random.shuffle(ind)
    _rate = tf.gather(_rate, rand_ind)
    sorted_rate = tf.sort(_rate)

    u = target_rate - sorted_rate
    tau = (tf.cast(tf.range(target_rate.shape[0]), tf.float32) + 1) / target_rate.shape[0]
    loss = huber_quantile_loss(u, tau, .002)

    return loss


class SpikeRateDistributionRegularization:
    def __init__(self, target_rates, rate_cost=.5):
        self._rate_cost = rate_cost
        self._target_rates = target_rates

    def __call__(self, spikes):
        reg_loss = compute_spike_rate_distribution_loss(spikes, self._target_rates) * self._rate_cost
        reg_loss = tf.reduce_sum(reg_loss)

        return reg_loss


class VoltageRegularization:
    def __init__(self, cell, voltage_cost=1e-5):
        self._voltage_cost = voltage_cost
        self._cell = cell

    def __call__(self, voltages):
        voltage_32 = (tf.cast(voltages, tf.float32) - self._cell.voltage_offset) / self._cell.voltage_scale
        v_pos = tf.square(tf.nn.relu(voltage_32 - 1.))
        v_neg = tf.square(tf.nn.relu(-voltage_32 + 1.))
        voltage_loss = tf.reduce_mean(tf.reduce_sum(v_pos + v_neg, -1)) * self._voltage_cost
        return voltage_loss


class SpikeVoltageRegularization(tf.keras.layers.Layer):
    def __init__(self, cell, rate_cost=.1, voltage_cost=.01, target_rate=.02):
        self._rate_cost = rate_cost
        self._voltage_cost = voltage_cost
        self._target_rate = target_rate
        self._cell = cell
        super().__init__()

    def call(self, inputs, **kwargs):
        spike = inputs[0]
        voltage = inputs[1]
        # upper_threshold = self._cell.threshold
        # if 'a_buf' in inputs[2].keys():
        #     upper_threshold += self._cell.beta[:, None, None, :] * inputs[2]['a_buf']

        rate = tf.reduce_mean(tf.cast(spike, tf.float32), axis=(0, 1))
        global_rate = tf.reduce_mean(rate)
        self.add_metric(global_rate, name='rate', aggregation='mean')

        reg_loss = tf.reduce_sum(tf.square(rate - self._target_rate)) * self._rate_cost
        self.add_loss(reg_loss)
        self.add_metric(reg_loss, name='rate_loss', aggregation='mean')

        voltage_32 = tf.cast(voltage, tf.float32)
        v_th_32 = tf.cast(self._cell.v_th, tf.float32)
        v_reset_32 = tf.cast(self._cell.v_reset, tf.float32)
        diff = v_th_32 - v_reset_32
        v_pos = tf.square(tf.clip_by_value(tf.nn.relu(voltage_32 - v_th_32), 0., 1.))
        v_neg = tf.square(tf.clip_by_value(tf.nn.relu(-voltage_32 + v_reset_32 - diff), 0., 1.))
        voltage_loss = tf.reduce_mean(tf.reduce_sum(v_pos + v_neg, -1)) * self._voltage_cost
        self.add_loss(voltage_loss)
        self.add_metric(voltage_loss, name='voltage_loss', aggregation='mean')
        return inputs
