import os
import numpy as np
import pickle as pkl
import tensorflow as tf
import matplotlib.pyplot as plt

import models
import toolkit

class NetworkVisualizer(tf.keras.callbacks.Callback):
    def __init__(self, test_data_set, extractor_model, cell, loss, readout_neuron_ids=None,
                 batch_ind=0, interactive=False, path=None, prefix='', dump_data=None):
        super().__init__()
        test_iter = iter(test_data_set)
        self._test_example = next(test_iter)
        self._extractor_model = extractor_model
        self._batch_ind = batch_ind
        self._cell = cell
        self._loss = loss
        self._interactive = interactive
        self._readout_neuron_ids = readout_neuron_ids
        self.fig, self.axes = plt.subplots(5, figsize=(8, 10), sharex=True)
        self.w_fig_e, self.w_ax_e = plt.subplots(figsize=(5, 5))
        self.w_fig_i, self.w_ax_i = plt.subplots(figsize=(5, 5))
        self._counter = 0
        self._path = path
        self._dump_data = dump_data
        self._prefix = prefix

    def on_epoch_begin(self, epoch, logs=None):
        inputs = self._test_example[0]
        targets = self._test_example[1]
        with tf.GradientTape() as tape:
            tape.watch(inputs[1])
            (z, v), prediction, all_prediction = self._extractor_model(inputs)
            loss = self._loss(targets, prediction)
        z_grad = tape.gradient(loss, inputs[1])

        [ax.clear() for ax in self.axes]

        times, inds = np.where(inputs[0][self._batch_ind].numpy() > .5)
        self.axes[0].plot(times, inds, 'k.', ms=1, alpha=.7)
        self.axes[0].set_ylim([0, inputs[0].shape[-1]])

        n_neurons = z.shape[-1]
        plot_ind_to_ind = np.arange(n_neurons)
        output_stripes = []
        n_outputs = self._readout_neuron_ids.shape[0]
        neurons_per_output = self._readout_neuron_ids.shape[1]
        st = int(n_neurons / n_outputs / 2)
        for _i in range(n_outputs):
            _t = plot_ind_to_ind[st:st + neurons_per_output].copy()
            plot_ind_to_ind[st:st + neurons_per_output] = self._readout_neuron_ids[_i]
            plot_ind_to_ind[self._readout_neuron_ids[_i]] = _t
            output_stripes.append(st)
            st += int(n_neurons / n_outputs)
        inverse_plot_ind_to_ind = np.zeros_like(plot_ind_to_ind)
        inverse_plot_ind_to_ind[plot_ind_to_ind] = np.arange(n_neurons)

        times, inds = np.where(z[self._batch_ind].numpy() > .5)

        self.axes[1].plot(times, inverse_plot_ind_to_ind[inds], 'r.', ms=1, alpha=.2)
        if self._readout_neuron_ids is not None:
            color = ['b', 'g', 'c', 'm', 'y', 'orange', 'lime', 'lightcoral', 'deeppink', 'slategray']
            for j in range(n_outputs):
                sel = np.zeros(inds.shape, np.bool)
                for _i in self._readout_neuron_ids[j]:
                    sel = np.logical_or(sel, inds == _i)
                self.axes[1].plot(times[sel], inverse_plot_ind_to_ind[inds[sel]],
                                  '.', color=color[j], ms=1, alpha=.5)
        # self.axes[1].set_ylim([-100, z.shape[-1]])

        np_grads = z_grad[self._batch_ind, ..., 2:].numpy()
        abs_max = np.percentile(np.abs(np_grads), 95)
        p = self.axes[2].pcolormesh(np_grads.T, cmap='seismic', vmin=-abs_max, vmax=abs_max)
        toolkit.do_inset_colorbar(self.axes[2], p, '', loc='middle')
        self.axes[2].set_ylim([0, z.shape[-1]])

        _v = v[self._batch_ind, :, :4].numpy()
        self.axes[3].plot(_v, alpha=.5)
        self.axes[3].set_ylim([-100, -25])

        filtered_all_prediction = models.exp_convolve(all_prediction[self._batch_ind], decay=.7, axis=0)
        self.axes[4].plot(tf.nn.softmax(filtered_all_prediction).numpy())
        self.axes[4].set_ylim([0, 1])

        self.w_ax_e.clear()
        self.w_ax_i.clear()
        rec_w = self.model.get_layer('rsnn').cell.recurrent_weight_values.numpy()
        sq = int(np.sqrt(rec_w.shape[0]))
        rec_w = rec_w[:sq**2].reshape((sq, sq))

        p = self.w_ax_e.pcolormesh(np.clip(rec_w, 0, np.inf), cmap='cividis')
        toolkit.do_inset_colorbar(self.w_ax_e, p, '')

        p = self.w_ax_i.pcolormesh(np.clip(-rec_w, 0, np.inf), cmap='cividis')
        toolkit.do_inset_colorbar(self.w_ax_i, p, '')

        # self.axes[3].plot(predictions[3][self._batch_ind, :, 0].numpy(), 'b', alpha=.9)
        # self.axes[3].plot(predictions[3][self._batch_ind, :, 1].numpy(), 'r', alpha=.9)

        # g = grads[self._batch_ind, ..., 0].numpy()
        # abs_max = np.abs(g).max()
        # p = self.axes[4].pcolormesh(g.T, cmap='seismic', vmin=-abs_max, vmax=abs_max)
        # toolkit.do_inset_colorbar(self.axes[4], p, '')
        # g = grads[self._batch_ind, ..., 1].numpy()
        # abs_max = np.abs(g).max()
        # p = self.axes[5].pcolormesh(g.T, cmap='seismic', vmin=-abs_max, vmax=abs_max)
        # toolkit.do_inset_colorbar(self.axes[5], p, '')
        # g = grads[self._batch_ind, ..., 2].numpy()
        # abs_max = np.abs(g).max()
        # with open('debug.pkl', 'wb') as f:
        #     pkl.dump(dict(g=g, v=predictions[1].numpy()), f)
        # print(g)
        # p = self.axes[6].pcolormesh(g.T, cmap='seismic', vmin=-abs_max, vmax=abs_max)
        # toolkit.do_inset_colorbar(self.axes[6], p, '')

        if self._interactive:
            plt.draw()
            plt.pause(.1)
        else:
            self.fig.savefig('interim_plot.png', dpi=300)
            self.w_fig_e.savefig('interim_w_e_plot.png', dpi=300)
            self.w_fig_i.savefig('interim_w_i_plot.png', dpi=300)
            self._counter += 1
            if self._path is not None:
                self.fig.savefig(os.path.join(self._path, f'{self._prefix}raster_epoch_{self._counter}.png'), dpi=300)
                self.w_fig_e.savefig(os.path.join(self._path, f'{self._prefix}rec_weights_e_{self._counter}.png'), dpi=300)
                self.w_fig_i.savefig(os.path.join(self._path, f'{self._prefix}rec_weights_i_{self._counter}.png'), dpi=300)
                if self._dump_data is not None and self._dump_data:
                    data = dict(inputs=inputs[0].numpy(), targets=targets.numpy(),
                                z=z.numpy(), v=v.numpy(), prediction=prediction.numpy(),
                                all_prediction=all_prediction.numpy(), loss=loss.numpy(),
                                z_grad=z_grad.numpy(), readout_neuron_ids=self._readout_neuron_ids)
                    with open(os.path.join(self._path, f'{self._prefix}dump_{self._counter}.pkl'), 'wb') as f:
                        pkl.dump(data, f)


class StopAt(tf.keras.callbacks.Callback):
    def __init__(self, key='val_accuracy', limit=.95):
        super().__init__()
        self._key = key
        self._limit = limit

    def on_epoch_end(self, epoch, logs=None):
        test_accuracy = logs.get(self._key)
        if test_accuracy > self._limit:
            self.model.stop_training = True
