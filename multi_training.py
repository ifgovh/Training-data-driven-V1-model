import matplotlib
matplotlib.use('agg')# to avoid GUI request on clusters
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import socket
import absl
import json
import time
import contextlib
import datetime as dt
import numpy as np
import pickle as pkl
import tensorflow as tf

import load_sparse
import classification_tools
import toolkit
import stim_dataset
import simmanager
import models


def main(_):
    flags = absl.app.flags.FLAGS
    results_dir = os.path.join(flags.results_dir, 'multi_training')
    os.makedirs(results_dir, exist_ok=True)

    per_replica_batch_size = flags.batch_size
    n_input = 17400

    # load firing rates
    with open(os.path.join(flags.data_dir, 'garrett_firing_rates.pkl'), 'rb') as f:
        firing_rates = pkl.load(f)
    sorted_firing_rates = np.sort(firing_rates)
    percentiles = (np.arange(firing_rates.shape[-1]) + 1).astype(np.float32) / firing_rates.shape[-1]
    rate_rd = np.random.RandomState(seed=flags.seed)
    x_rand = rate_rd.uniform(size=flags.neurons)
    target_firing_rates = np.sort(np.interp(x_rand, percentiles, sorted_firing_rates))

    # physical_devices = tf.config.list_physical_devices('GPU')
    # try:
    #     for dev in physical_devices:
    #         tf.config.experimental.set_memory_growth(dev, True)
    # except:
    #     # Invalid device or cannot modify virtual devices once initialized.
    #     pass

    dtype = tf.float32
    
    if socket.gethostname().count('nvcluster') > 0 or socket.gethostname().count('pCluster') > 0:
        n_workers, task_id = 1, 0
        n_gpus_per_worker = 1
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice(reduce_to_device="cpu:0"))
    else:
        n_workers, task_id = toolkit.set_tf_config_from_slurm(port=flags.port)
        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
            communication=tf.distribute.experimental.CollectiveCommunication.NCCL)
        n_gpus_per_worker = int(strategy.num_replicas_in_sync / n_workers)
        
    is_master = task_id < .5
    print(f'worker {task_id + 1} / {n_workers}')

    # tasks = ['garrett', 'evidence', 'vcd_grating', 'ori_diff', '10class']

    # for uneven batch distribution
    if task_id < 8:
        task_name = 'garrett'
    elif task_id < 16:
        task_name = 'evidence'
    elif task_id < 24:
        task_name = 'vcd_grating'
    elif task_id < 32:
        task_name = 'ori_diff'
    elif task_id < 40:
        task_name = '10class'

    # task_name = tasks[task_id % len(tasks)] # this for even distribution

    global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
    if task_name == '10class':
        n_output = 10
    else:
        n_output = 2 #although the garrat and vcd_grating, ori_diff only need one readout population but it has another pesudo ouput (thresh)

    # load column model of Billeh et al
    if flags.caching:
        load_fn = load_sparse.cached_load_billeh
    else:
        load_fn = load_sparse.load_billeh
    input_population, network, bkg_weights = load_fn(
        n_input=n_input, n_neurons=flags.neurons, core_only=flags.core_only, data_dir=flags.data_dir,
        seed=flags.seed, connected_selection=flags.connected_selection, n_output=n_output,
        neurons_per_output=flags.neurons_per_output, use_rand_ini_w=flags.use_rand_ini_w,
        use_dale_law=flags.use_dale_law, use_only_one_type=flags.use_only_one_type,
        use_rand_connectivity=flags.use_rand_connectivity, scale_w_e=flags.scale_w_e,
        localized_readout=flags.localized_readout, use_uniform_neuron_type=flags.use_uniform_neuron_type)

    noise_scales = [float(a) for a in flags.scale.split(',') if a != '']

    with strategy.scope():
        model = classification_tools.create_model(
            network, input_population, bkg_weights, seq_len=flags.seq_len, n_input=n_input,
            n_output=n_output, dtype=dtype,
            input_weight_scale=flags.input_weight_scale,
            dampening_factor=flags.dampening_factor, gauss_std=flags.gauss_std,            
            train_recurrent=flags.train_recurrent,
            train_input=flags.train_input, lRout_pop='all', use_decoded_noise=flags.use_decoded_noise,
            neuron_output=flags.neuron_output, L2_factor=0, return_state=True,
            max_delay=flags.max_delay, batch_size=flags.batch_size,            
            output_mode=task_name, down_sampled_decode_noise_path=os.path.join(flags.data_dir, 'additive_noise.mat'),
            neuron_model=flags.neuron_model, use_dale_law=flags.use_dale_law, scale=noise_scales,            
        )

        model.build((flags.batch_size, flags.seq_len, n_input))
        rsnn_layer = model.get_layer('rsnn')
        rec_weight_regularizer = models.StiffRegularizer(flags.recurrent_weight_regularization,
                                                         rsnn_layer.cell.recurrent_weight_values)

        rate_distribution_regularizer = models.SpikeRateDistributionRegularization(target_firing_rates, flags.rate_cost)

        prediction_layer = model.get_layer('prediction')
        extractor_model = tf.keras.Model(inputs=model.inputs,
                                         outputs=[rsnn_layer.output, model.output, prediction_layer.output])
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(_l, _p, _w):
            per_example_loss = loss_object(_l, _p, sample_weight=_w) * strategy.num_replicas_in_sync / tf.reduce_sum(_w)
            rec_weight_loss = rec_weight_regularizer(rsnn_layer.cell.recurrent_weight_values)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size) + rec_weight_loss

        
        optimizer = tf.keras.optimizers.Adam(flags.learning_rate, epsilon=1e-11)    

    def get_dataset_fn(is_test=False, _steps_per_epoch=80):
        def _f(input_context):
            if task_name == 'garrett':
                if is_test:
                    path = os.path.join(flags.data_dir, '../alternate_small_stimuli.pkl')
                    n_images = 8
                else:
                    path = os.path.join(flags.data_dir, '../many_small_stimuli.pkl')
                    n_images = 40

                delays = [int(a) for a in flags.delays.split(',') if a != '']
                if flags.from_lgn:
                    _data_set = stim_dataset.generate_data_set_continuing(
                        path, seq_len=flags.seq_len, batch_size=per_replica_batch_size * n_gpus_per_worker,
                        examples_in_epoch=int(2 * _steps_per_epoch * flags.seq_len / np.min(delays)),
                        p_reappear=flags.p_reappear, n_images=n_images, current_input=flags.current_input,
                        im_slice=flags.im_slice, delay=delays[0]).unbatch().batch(per_replica_batch_size).prefetch(1)
                else:
                    _data_set = stim_dataset.generate_VCD_NI_from_path(path=os.path.join(flags.data_dir, '../575_train_img_100x174.h5'),
                        intensity=flags.sti_intensity, im_slice=flags.im_slice, pre_delay=50, post_delay=150, p_reappear=0.5,
                                          from_lgn=False, pre_chunks=4, resp_chunks=1, post_chunks=1, current_input=True).batch(per_replica_batch_size).prefetch(1)

            elif task_name == 'vcd_grating':
                # seq_len must be 600 to use this
                _data_set = stim_dataset.generate_VCD_orientation(from_lgn=flags.from_lgn, intensity=flags.sti_intensity, im_slice=100, pre_delay=50, post_delay=150, p_reappear=flags.p_reappear, current_input=True)
                _data_set = _data_set.batch(per_replica_batch_size).prefetch(1)

            elif task_name == 'evidence':
                if flags.from_lgn:
                    _data_set = stim_dataset.generate_evidence_accumulation_via_LGN(file_name=os.path.join(flags.data_dir, '../EA_LGN.h5'), seq_len=flags.seq_len, pause=250, n_cues=5, cue_len=50, interval_len=10, recall_len=50)
                    _data_set = _data_set.batch(per_replica_batch_size).prefetch(1)
                else:
                    path = os.path.join(flags.data_dir, '../evidence_accumulation_data.pkl')
                    _data_set = stim_dataset.generate_evidence_accumulation(path, batch_size=per_replica_batch_size, seq_len=flags.seq_len,
                        n_examples_per_epoch=int(global_batch_size / n_workers) * _steps_per_epoch).batch(per_replica_batch_size).prefetch(1)

            elif task_name == 'ori_diff':
                _data_set = stim_dataset.generate_fine_orientation_discrimination(from_lgn=flags.from_lgn, intensity=flags.sti_intensity, im_slice=flags.im_slice, pre_delay=flags.pre_delay, post_delay=flags.post_delay,
                                                         pre_chunks=flags.pre_chunks, resp_chunks=1, post_chunks=flags.post_chunks, current_input=True).batch(per_replica_batch_size).prefetch(1)
            elif task_name == '10class':
                if is_test:
                    n_examples = 9984
                else:
                    n_examples = 49984 #int(50000/64)

                _data_set = stim_dataset.generate_pure_classification_data_set_from_generator(
                    data_usage=int(is_test),intensity=flags.sti_intensity,im_slice=flags.im_slice,
                    pre_delay=flags.pre_delay, post_delay=flags.post_delay, current_input=flags.current_input,
                    dataset='mnist', pre_chunks=flags.pre_chunks, resp_chunks=1, from_lgn=flags.from_lgn,
                    post_chunks=flags.post_chunks).take(n_examples).batch(per_replica_batch_size).shard(8, input_context.input_pipeline_id - 32).prefetch(1) # task_id = input_context.input_pipeline_id, [16,23] is 10class
                    # post_chunks=flags.post_chunks).take(n_examples).batch(per_replica_batch_size).shard(8, input_context.input_pipeline_id//3).prefetch(8) # 49984=int(50000/64); 8 nodes for each task, so divide it to 8 parts; total 3 tasks, so every 3 task_ids will choose the correct part
            return _data_set
        return _f


    zero_state = rsnn_layer.cell.zero_state(flags.batch_size)
    with strategy.scope():
        state_variables = tf.nest.map_structure(lambda a: tf.Variable(
            a, trainable=False, synchronization=tf.VariableSynchronization.ON_READ
        ), zero_state)

        train_loss = tf.keras.metrics.Mean()
        val_loss = tf.keras.metrics.Mean()
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        train_firing_rate = tf.keras.metrics.Mean()
        val_firing_rate = tf.keras.metrics.Mean()

        train_rate_loss = tf.keras.metrics.Mean()
        val_rate_loss = tf.keras.metrics.Mean()
        train_voltage_loss = tf.keras.metrics.Mean()
        val_voltage_loss = tf.keras.metrics.Mean()

        def reset_train_metrics():
            train_loss.reset_states(), train_accuracy.reset_states(), train_firing_rate.reset_states()
            train_rate_loss.reset_states(), train_voltage_loss.reset_states()

        def reset_validation_metrics():
            val_loss.reset_states(), val_accuracy.reset_states(), val_firing_rate.reset_states()
            val_rate_loss.reset_states(), val_voltage_loss.reset_states()

    def roll_out(_x, _y, _w):
        _initial_state = tf.nest.map_structure(lambda _a: _a.read_value(), state_variables)
        _out, _p, _ = extractor_model((_x, _initial_state))
        _z, _v = _out[0]
        voltage_32 = (tf.cast(_v, tf.float32) - rsnn_layer.cell.voltage_offset) / rsnn_layer.cell.voltage_scale
        v_pos = tf.square(tf.nn.relu(voltage_32 - 1.))
        v_neg = tf.square(tf.nn.relu(-voltage_32 + 1.))
        voltage_loss = tf.reduce_mean(tf.reduce_sum(v_pos + v_neg, -1)) * flags.voltage_cost
        rate_loss = rate_distribution_regularizer(_z)
        classification_loss = compute_loss(_y, _p, _w)
        _aux = dict(rate_loss=rate_loss, voltage_loss=voltage_loss)
        _loss = classification_loss + rate_loss + voltage_loss
        return _out, _p, _loss, _aux

    def train_step(_x, _y, _w):
        with tf.GradientTape() as tape:
            _out, _p, _loss, _aux = roll_out(_x, _y, _w)

        _op = train_accuracy.update_state(_y, _p, sample_weight=_w)
        with tf.control_dependencies([_op]):
            _op = train_loss.update_state(_loss)
        _rate = tf.reduce_mean(_out[0][0])
        with tf.control_dependencies([_op]):
            _op = train_firing_rate.update_state(_rate)
        with tf.control_dependencies([_op]):
            _op = train_rate_loss.update_state(_aux['rate_loss'])
        with tf.control_dependencies([_op]):
            _op = train_voltage_loss.update_state(_aux['voltage_loss'])

        grad = tape.gradient(_loss, model.trainable_variables)
        for g, v in zip(grad, model.trainable_variables):
            with tf.control_dependencies([_op]):
                _op = optimizer.apply_gradients([(g, v)])

    @tf.function
    def distributed_train_step(_x, _y, _w):
        strategy.run(train_step, args=(_x, _y, _w))

    def train_step_continuing(_x, _y, _w):
        with tf.GradientTape() as tape:
            _out, _p, _loss, _aux = roll_out(_x, _y, _w)

        _op = train_accuracy.update_state(_y, _p, sample_weight=_w)
        with tf.control_dependencies([_op]):
            _op = train_loss.update_state(_loss)
        _rate = tf.reduce_mean(_out[0][0])
        with tf.control_dependencies([_op]):
            _op = train_firing_rate.update_state(_rate)
        with tf.control_dependencies([_op]):
            _op = train_rate_loss.update_state(_aux['rate_loss'])
        with tf.control_dependencies([_op]):
            _op = train_voltage_loss.update_state(_aux['voltage_loss'])
        tf.nest.map_structure(lambda _a, _b: _a.assign(_b), list(state_variables), _out[1:])
        # grad = tape.gradient(_loss, model.trainable_variables)
        # optimizer.apply_gradients(zip(grad, model.trainable_variables))
        grad = tape.gradient(_loss, model.trainable_variables)
        for g, v in zip(grad, model.trainable_variables):
            with tf.control_dependencies([_op]):
                _op = optimizer.apply_gradients([(g, v)])

    @tf.function
    def distributed_train_step_continuing(_x, _y, _w):
        strategy.run(train_step_continuing, args=(_x, _y, _w))

    def validation_step(_x, _y, _w):
        _out, _p, _loss, _aux = roll_out(_x, _y, _w)
        _op = val_accuracy.update_state(_y, _p, sample_weight=_w)
        with tf.control_dependencies([_op]):
            _op = val_loss.update_state(_loss)
        _rate = tf.reduce_mean(_out[0][0])
        with tf.control_dependencies([_op]):
            _op = val_firing_rate.update_state(_rate)
        with tf.control_dependencies([_op]):
            _op = val_rate_loss.update_state(_aux['rate_loss'])
        with tf.control_dependencies([_op]):
            _op = val_voltage_loss.update_state(_aux['voltage_loss'])
        # tf.nest.map_structure(lambda _a, _b: _a.assign(_b), list(state_variables), _out[1:])

    @tf.function
    def distributed_validation_step(_x, _y, _w):
        strategy.run(validation_step, args=(_x, _y, _w))

    def validation_step_continuing(_x, _y, _w):
        _out, _p, _loss, _aux = roll_out(_x, _y, _w)
        _op = val_accuracy.update_state(_y, _p, sample_weight=_w)
        with tf.control_dependencies([_op]):
            _op = val_loss.update_state(_loss)
        _rate = tf.reduce_mean(_out[0][0])
        with tf.control_dependencies([_op]):
            _op = val_firing_rate.update_state(_rate)
        with tf.control_dependencies([_op]):
            _op = val_rate_loss.update_state(_aux['rate_loss'])
        with tf.control_dependencies([_op]):
            _op = val_voltage_loss.update_state(_aux['voltage_loss'])
        tf.nest.map_structure(lambda _a, _b: _a.assign(_b), list(state_variables), _out[1:])

    @tf.function
    def distributed_validation_step_continuing(_x, _y, _w):
        strategy.run(validation_step_continuing, args=(_x, _y, _w))

    def reset_state():
        tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, zero_state)

    @tf.function
    def distributed_reset_state():
        strategy.run(reset_state)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    if task_id in [0,8,16,24,32]:
        sim_name = toolkit.get_random_identifier('b_')
        print(f'> Results for {task_name} will be stored in {os.path.join(results_dir, sim_name)}')

        cm = simmanager.SimManager(sim_name, results_dir, write_protect_dirs=False, tee_stdx_to='output.log')
    else:
        cm = contextlib.nullcontext()

    if flags.restore_from != '':
        with strategy.scope():
            # checkpoint.restore(tf.train.latest_checkpoint(flags.restore_from))
            checkpoint.restore(flags.restore_from)
            print(f'Model parameters of {task_name} restored from {flags.restore_from}')

    def compose_str(_loss, _acc, _rate, _rate_loss, _voltage_loss):
        _s = f'Loss {_loss:.4f}, '
        _s += f'RLoss {_rate_loss:.4f}, '
        _s += f'VLoss {_voltage_loss:.4f}, '
        _s += f'Accuracy {_acc:.4f}, '
        _s += f'Rate {_rate:.4f}'
        return _s


    test_data_set = strategy.experimental_distribute_datasets_from_function(
        get_dataset_fn(True, flags.val_steps))
    train_data_set = strategy.experimental_distribute_datasets_from_function(
        get_dataset_fn(False, flags.steps_per_epoch))

    with cm:
        if is_master:
            step_counter = tf.Variable(0,trainable=False)
            manager = tf.train.CheckpointManager(
                checkpoint, directory=cm.paths.results_path, max_to_keep=100,
                # keep_checkpoint_every_n_hours=2,
                checkpoint_interval = 1, # save ckpt for data analysis
                step_counter=step_counter
            )
            summary_writer = tf.summary.create_file_writer(cm.paths.results_path)

            def save_model():
                if is_master:
                    step_counter.assign_add(1)
                    p = manager.save()
                    print(f'Model saved in {p}')

            with open(os.path.join(cm.paths.data_path, 'config.json'), 'w') as f:
                json.dump(flags.flag_values_dict(), f, indent=4)

        stop = False
        t0 = time.time()
        
        for epoch in range(flags.n_epochs):
            if stop:
                break

            it = iter(train_data_set)
            date_str = dt.datetime.now().strftime('%d-%m-%Y %H:%M')
            print(f'Epoch {epoch + 1:2d}/{flags.n_epochs} @ {date_str}')
            # quit()
            distributed_reset_state()
            for step in range(flags.steps_per_epoch):
                x, y, _, w = next(it)
                if task_name == 'garrett' or task_name == 'vcd_grating':
                    distributed_train_step_continuing(x, y, w)
                else:
                    distributed_train_step(x, y, w)

                print_str = f'  Step {step + 1:2d}/{flags.steps_per_epoch}: '
                print_str += compose_str(train_loss.result(), train_accuracy.result(),
                                         train_firing_rate.result(), train_rate_loss.result(), train_voltage_loss.result())
                # write_csv(time.time(), task_id, step, train_loss.result(), train_accuracy.result(),
                                         # train_firing_rate.result(), train_rate_loss.result(), train_voltage_loss.result(), epoch, True, os.path.join(results_dir, 'new_5_tasks'))
                print(print_str, end='\r')
                if 0 < flags.max_time < (time.time() - t0) / 3600:
                    stop = True
                    break
            print()
            if stop:
                print(f'[ Maximum optimization time of {flags.max_time:.2f}h reached ]')

            distributed_reset_state()
            test_it = iter(test_data_set)
            for step in range(flags.val_steps):
                x, y, _, w = next(test_it)
                if task_name == 'garrett' or task_name == 'vcd_grating':
                    distributed_validation_step_continuing(x, y, w)
                else:
                    distributed_validation_step(x, y, w) 

            print_str = '  Validation: ' + compose_str(
                val_loss.result(), val_accuracy.result(), val_firing_rate.result(),
                val_rate_loss.result(), val_voltage_loss.result())
            
            print(print_str)
            keys = ['train_accuracy', 'train_loss', 'train_firing_rate', 'train_rate_loss',
                    'train_voltage_loss', 'val_accuracy', 'val_loss',
                    'val_firing_rate', 'val_rate_loss', 'val_voltage_loss']
            values = [a.result().numpy() for a in [train_accuracy, train_loss, train_firing_rate, train_rate_loss,
                                                   train_voltage_loss, val_accuracy, val_loss, val_firing_rate,
                                                   val_rate_loss, val_voltage_loss]]
            if stop:
                result = dict(
                    train_loss=float(train_loss.result().numpy()),
                    train_accuracy=float(train_accuracy.result().numpy()),
                    test_loss=float(val_loss.result().numpy()),
                    test_accuracy=float(val_accuracy.result().numpy())
                )
            if is_master:
                save_model()
                with summary_writer.as_default():
                    for k, v in zip(keys, values):
                        tf.summary.scalar(k, v, step=epoch)
            if is_master and stop:
                with open(os.path.join(cm.paths.results_path, 'result.json'), 'w') as f:
                    json.dump(result, f)
            reset_train_metrics()
            reset_validation_metrics()


if __name__ == '__main__':
    hostname = socket.gethostname()
    if hostname.count('scherr-pc') > 0:
        _data_dir = '/data/allen/v1_model/GLIF_network'
        _results_dir = '/data/output/billeh'
    elif hostname.count('nvcluster') > 0:
        _data_dir = os.path.expanduser('~/tf_billeh_column/GLIF_network')
        _results_dir = '/srv/local/ifgovh/RESULTS'
    elif hostname.count('juwels') > 0:
        _data_dir = '/p/project/structuretofunction/guozhang/glif_criticality/GLIF_network'
        _results_dir = '/p/scratch/structuretofunction/chen/RESULTS'
    elif hostname.count('pCluster') > 0:
        # _data_dir = '/calc/scherr/allen/v1_model/GLIF_network'
        # _results_dir = '/calc/scherr/output/billeh'
        _data_dir = '/home/guozhang/tf_billeh_column/GLIF_network'
        _results_dir = '/home/guozhang/RESULTS'
    elif hostname.count('nid') > 0:
        _data_dir = '/users/bp000436/glif_criticality/GLIF_network'
        _results_dir = '/scratch/snx3000/bp000436/RESULTS'
    else:
        _data_dir = ''
        _results_dir = ''

    absl.app.flags.DEFINE_string('data_dir', _data_dir, '')
    absl.app.flags.DEFINE_string('results_dir', _results_dir, '')
    absl.app.flags.DEFINE_string('restore_from', '', '')
    absl.app.flags.DEFINE_string('comment', '', '')
    absl.app.flags.DEFINE_string('delays', '200,200', '')
    absl.app.flags.DEFINE_string('neuron_model', 'GLIF3', '')
    absl.app.flags.DEFINE_string('scale', '2,2', '')

    absl.app.flags.DEFINE_float('learning_rate', .001, '')
    absl.app.flags.DEFINE_float('rate_cost', .1, '')
    absl.app.flags.DEFINE_float('voltage_cost', .00001, '')
    absl.app.flags.DEFINE_float('dampening_factor', .5, '')
    absl.app.flags.DEFINE_float('gauss_std', .28, '')
    absl.app.flags.DEFINE_float('recurrent_weight_regularization', 0., '')
    absl.app.flags.DEFINE_float('p_reappear', .5, '')
    absl.app.flags.DEFINE_float('max_time', -1, '')
    absl.app.flags.DEFINE_float('scale_w_e', -1, '')
    absl.app.flags.DEFINE_float('sti_intensity', 2., '')
    absl.app.flags.DEFINE_float('input_weight_scale', 1., '')

    absl.app.flags.DEFINE_integer('n_epochs', 1000, '')
    absl.app.flags.DEFINE_integer('batch_size', 2, '')
    absl.app.flags.DEFINE_integer('neurons', 51978, '')
    absl.app.flags.DEFINE_integer('seq_len', 600, '')
    absl.app.flags.DEFINE_integer('im_slice', 100, '')
    absl.app.flags.DEFINE_integer('seed', 3000, '')
    absl.app.flags.DEFINE_integer('port', 12778, '')
    absl.app.flags.DEFINE_integer('neurons_per_output', 30, '')
    absl.app.flags.DEFINE_integer('steps_per_epoch', 781, '')# EA and garret dose not need this many but pure classification needs 781 = int(50000/64)
    absl.app.flags.DEFINE_integer('val_steps', 156, '')# EA and garret dose not need this many but pure classification needs 156 = int(10000/64)
    absl.app.flags.DEFINE_integer('max_delay', 5, '')
    absl.app.flags.DEFINE_integer('n_plots', 1, '')

    absl.app.flags.DEFINE_integer('pre_chunks', 3, '')
    absl.app.flags.DEFINE_integer('post_chunks', 8, '') # the pure calssification task only need 1 but to make consistent with other tasks one has to make up here
    absl.app.flags.DEFINE_integer('pre_delay', 50, '')
    absl.app.flags.DEFINE_integer('post_delay', 450, '')

    absl.app.flags.DEFINE_boolean('use_rand_connectivity', False, '')
    absl.app.flags.DEFINE_boolean('use_uniform_neuron_type', False, '')
    absl.app.flags.DEFINE_boolean('use_only_one_type', False, '')
    absl.app.flags.DEFINE_boolean('use_dale_law', True, '')
    absl.app.flags.DEFINE_boolean('caching', False, '') # if one wants to use caching, remember to update the caching function
    absl.app.flags.DEFINE_boolean('core_only', False, '')
    absl.app.flags.DEFINE_boolean('train_input', True, '')
    absl.app.flags.DEFINE_boolean('train_recurrent', True, '')
    absl.app.flags.DEFINE_boolean('connected_selection', True, '')
    absl.app.flags.DEFINE_boolean('neuron_output', True, '')
    absl.app.flags.DEFINE_boolean('localized_readout', True, '')
    absl.app.flags.DEFINE_boolean('current_input', True, '')
    absl.app.flags.DEFINE_boolean('use_rand_ini_w', False, '')
    absl.app.flags.DEFINE_boolean('use_decoded_noise', True, '')
    absl.app.flags.DEFINE_boolean('from_lgn', True, '')

    absl.app.run(main)

