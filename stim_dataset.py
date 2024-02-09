import os
import h5py
import numpy as np
import pickle as pkl
import tensorflow as tf
import pdb
import lgn_model

def load_firing_rates(path='image_outputs'):
    with open(path, 'rb') as f:
        d = pkl.load(f)
    try: # Franz's stuff
        # inds = [0, 3, 5]
        # rates = np.stack(list(d.values()))[inds]
        rates = np.stack(list(d.values()))
    except:
        rates = d
    return rates


def load_firing_rates_tf(path):
    with tf.device('/cpu'):
        np_rates = load_firing_rates(path).astype(np.float32)
        rates = tf.Variable(np_rates, trainable=False)
    return rates


def generate_pair(n_total, p_reappear=.1):
    first_index = tf.cast(tf.random.uniform(()) * n_total, tf.int32)
    logits = -tf.one_hot(first_index, n_total) * 1e9
    second_index = tf.where(
        tf.random.uniform(()) > p_reappear,
        tf.cast(tf.random.categorical([logits], 1)[0, 0], tf.int32),
        first_index)
    return first_index, second_index


def remove_first_dim(_x):
    t_shp = _x.shape
    tf_shp = tf.shape(_x)
    shp = []
    for i in range(len(t_shp)):
        shp.append(t_shp[i] if t_shp[i] is not None else tf_shp[i])
    new_shp = [shp[0] * shp[1], *shp[2:]]
    return tf.reshape(_x, new_shp)


def switch_time_and_batch(_x):
    perm = np.arange(len(_x.shape))
    perm[:2] = perm[:2][::-1]
    return tf.transpose(_x, perm)

def generate_data_set_continuing(path='image_outputs', batch_size=1, seq_len=1000, examples_in_epoch=50,
                                 p_reappear=.1, im_slice=250, delay=500, n_images=8, dtype=tf.float32,
                                 current_input=True, pre_chunks=4, resp_chunks=1):
    if path.split('.')[-1] == 'pkl':
        rates = load_firing_rates_tf(path)
    elif path.split('.')[-1] == 'h5':
        f = h5py.File(path, "r")
        rates = f["rates"][()]
        f.close()

    n_chunks = int((im_slice + delay) / 50)
    n_chunks_img = int(im_slice / 50)
    t_chunk = int((im_slice + delay) / n_chunks)
    assert n_chunks * 50 == im_slice + delay
    fixed_noise = tf.random.uniform(shape=(seq_len, batch_size, rates.shape[-1]))

    def concat(sub_seq, sub_label):
        _im = tf.gather(rates, sub_seq)[:, 50:im_slice + delay] # what is the structure of rates? [image index, LGN neuron, 50-ms gray + 100-ms image + 850-ms gray]
        _pause = tf.tile(rates[0, rates.shape[1] - 50:][None], (batch_size, 1, 1))
        _seq = tf.concat((_pause, _im), 1)
        _seq = tf.transpose(_seq, (1, 0, 2))

        _seq = tf.reshape(_seq, (n_chunks, t_chunk, batch_size, -1)) # what are n_chunks, t_chunk for? t_chunk=50 ms, n_chunks is the pesudo-batch; it would be cut to slices by the second unbatch(); then use the batch(700/5) to cancate and then multiply first two together.
        _tz = tf.zeros_like(sub_label)
        _label = tf.stack([_tz] * pre_chunks + [sub_label] * resp_chunks + [_tz] * (n_chunks - pre_chunks - resp_chunks))
        _img_label = tf.stack([_tz] + [sub_seq] * n_chunks_img + [_tz] * (n_chunks - n_chunks_img - 1))
        _tz = tf.zeros_like(sub_label, dtype=dtype) + .05
        _to = tf.ones_like(sub_label, dtype=dtype)
        _weights = tf.stack([_tz] * pre_chunks + [_to] * resp_chunks + [_tz] * (n_chunks - pre_chunks - resp_chunks)) #?? why not for whole response window but only for the onset 50 ms? the _seq is 50-ms dealy + 100-ms image + 150-ms delay; each trunk is 50 ms
        return _seq, _label, _img_label, _weights

    def gen_seq(_):
        # generate index for getting rate and label (diff or same)
        a = tf.TensorArray(tf.int32, size=2 * examples_in_epoch + 1)
        b = tf.TensorArray(tf.int32, size=2 * examples_in_epoch + 1)
        current_index = tf.cast(tf.random.uniform((batch_size,)) * n_images, tf.int32)

        a = a.write(0, current_index)
        b = b.write(0, tf.zeros((batch_size,), tf.int32))

        for i in tf.range(2 * examples_in_epoch):
            logits = -tf.one_hot(current_index, n_images) * 1e9

            change = tf.random.uniform((batch_size,)) > p_reappear
            new_index = tf.cast(tf.random.categorical(logits, 1)[:, 0], tf.int32)
            current_index = tf.where(
                change,
                new_index,
                current_index)
            a = a.write(i + 1, current_index)
            b = b.write(i + 1, tf.cast(change, tf.int32))
        sequences = tf.reshape(a.stack()[:-1], (examples_in_epoch * 2, batch_size))
        change = tf.reshape(b.stack()[:-1], (examples_in_epoch * 2, batch_size))
        return sequences, change

    def sample_poisson(_a):
        # assuming dt = 1 ms
        _p = 1 - tf.exp(-_a / 1000.)
        # _z = tf.cast(fixed_noise < _p, dtype)
        if current_input:
            _z = _p * 1.3
        else:
            _z = tf.cast(tf.random.uniform(tf.shape(_p)) < _p, dtype)
        return _z

    def l1(_seq, _l, _i, _w):
        return sample_poisson(remove_first_dim(_seq)), _l, _i, _w

    def l2(*_x):
        return tf.nest.map_structure(switch_time_and_batch, _x)
    # this batch is just to generate two image pair not the real batch
    data_set = tf.data.Dataset.from_tensor_slices([0]).map(gen_seq).unbatch().map(concat).unbatch().batch(
        int(seq_len / 50)).map(l1).map(l2)
    return data_set

def generate_VCD_NI_from_path(path, intensity=2, im_slice=100, pre_delay=50, post_delay=150, p_reappear=0.5,
                              pre_chunks=10, resp_chunks=1, post_chunks=1, current_input=True,
                              batch_size=2, pairs_in_epoch=781,from_lgn=True):
    # hard code lgn scale for the case from_lgn=False
    mimc_lgn_std, mimc_lgn_mean = 0.01254, 0.01140


    lgn = lgn_model.LGN()
    seq_len = (pre_delay + im_slice + post_delay)*2
    chunk_size = 50 # ms
    n_chunks = int(seq_len / chunk_size)

    assert n_chunks == 2*(resp_chunks + pre_chunks + post_chunks)

    f = h5py.File(path, "r")
    x_train = f["data"][()]
    f.close()

    num_imgs = x_train.shape[0]

    if len(x_train.shape) > 3:
        x_train = tf.image.rgb_to_grayscale(x_train) / 255
    else:
        x_train = x_train[...,None]/255

    changes = np.random.uniform(size=[batch_size*pairs_in_epoch,2]) > p_reappear
    for i in range(batch_size):
        changes[i*pairs_in_epoch,0] = 0 # the first one cannot change

    img_id_seq = []
    for i, change in enumerate(changes):
        if change[0]:
            new_id = np.random.choice(num_imgs, 1)
            while new_id == img_id_seq[-1]:
                new_id = np.random.choice(num_imgs, 1)
            img_id_seq.append(new_id)
        else:
            if i < 1:
                img_id_seq.append(np.random.choice(num_imgs, 1))
            else:
                img_id_seq.append(img_id_seq[-1])

        if change[1]:
            new_id = np.random.choice(num_imgs, 1)
            while new_id == img_id_seq[-1]:
                new_id = np.random.choice(num_imgs, 1)
            img_id_seq.append(new_id)
        else:
            if i < 1:
                img_id_seq.append(np.random.choice(num_imgs, 1))
            else:
                img_id_seq.append(img_id_seq[-1])

    img_id_seq = np.array(img_id_seq).reshape(batch_size, -1, 2)
    changes = changes.reshape(batch_size, -1, 2)
    # re-arrange for batches
    temp_ids = []
    temp_cha = []
    for i in range(batch_size):
        temp_ids.append(img_id_seq[i,...])
        temp_cha.append(changes[i,...])

    img_id_seq = np.concatenate(temp_ids, axis=1)
    changes = np.concatenate(temp_cha, axis=1)
    img_id_seq = img_id_seq.reshape(-1,2)
    changes = changes.reshape(-1,2)

    img_id_seq = tf.cast(tf.convert_to_tensor(img_id_seq), tf.float32)
    changes = tf.cast(tf.convert_to_tensor(changes), tf.float32)

    def gen_one_video(img_ind):
        if from_lgn:
            img = tf.image.resize_with_pad(x_train[tf.cast(img_ind,tf.int32)], 120, 240, method='lanczos5')
            tiled_img = tf.tile(img[None,...], (im_slice, 1, 1, 1))
            # make it in [-intensity, intensity]
            tiled_img = (tiled_img - .5) * intensity / .5
        else:
            # to mimic the 17400 dim of LGN output
            img = tf.image.resize_with_pad(x_train[tf.cast(img_ind,tf.int32)], 100, 174, method='lanczos5')
            # maintain the images for a while
            tiled_img = tf.tile(img[None,...], (im_slice, 1, 1, 1))

        # add an empty period before a period of real image for continuing classification
        z1 = tf.tile(tf.zeros_like(tiled_img[0,...])[None,...], (pre_delay, 1, 1, 1))
        z2 = tf.tile(tf.zeros_like(tiled_img[0,...])[None,...], (post_delay, 1, 1, 1))
        video = tf.concat((z1, tiled_img, z2), 0)
        return video, img_ind

    def _g():
        for change, img_id in zip(changes, img_id_seq):
            video1, img_id1 = gen_one_video(img_id[0])
            video2, img_id2 = gen_one_video(img_id[1])

            videos = tf.concat((video1, video2), 0)
            if from_lgn:
                spatial = lgn.spatial_response(videos)
                firing_rates = lgn.firing_rates_from_spatial(*spatial)
            else:
                firing_rates = tf.reshape(videos, [-1,17400])

            # sample rate
            # assuming dt = 1 ms
            _p = 1 - tf.exp(-firing_rates / 1000.)
            # _z = tf.cast(fixed_noise < _p, dtype)
            if current_input:
                _z = _p * 1.3
                if not from_lgn:
                    _z = _z * mimc_lgn_std
                    _z = (_z - tf.reduce_mean(_z)) / tf.math.reduce_std(_z) * mimc_lgn_std + mimc_lgn_mean
            else:
                _z = tf.cast(tf.random.uniform(tf.shape(_p)) < _p, tf.float32)

            ground_truth = tf.cast(change, tf.float32)
            label = tf.concat([tf.zeros(pre_chunks)] + [ground_truth[0]*tf.ones(resp_chunks)] + [tf.zeros(post_chunks)] +\
                [tf.zeros(pre_chunks)] + [ground_truth[1]*tf.ones(resp_chunks)] + [tf.zeros(post_chunks)],axis=0)
            weight = tf.concat([0.0*tf.ones(pre_chunks)] + [tf.ones(resp_chunks)] + [0.0*tf.ones(post_chunks)] +\
                               [0.0*tf.ones(pre_chunks)] + [tf.ones(resp_chunks)] + [0.0*tf.ones(post_chunks)], axis=0)
            # for plotting, label the image when it holds on
            image_label1 = tf.concat([tf.zeros(int(pre_delay/chunk_size))] + [img_id1*tf.ones(int(im_slice/chunk_size))] + [tf.zeros(int(post_delay/chunk_size))],axis=0)
            image_label2 = tf.concat([tf.zeros(int(pre_delay/chunk_size))] + [img_id2*tf.ones(int(im_slice/chunk_size))] + [tf.zeros(int(post_delay/chunk_size))],axis=0)
            image_labels = tf.concat([image_label1,image_label2],axis=0)
            yield _z, label, image_labels, weight


    output_dtypes = (tf.float32, tf.int32, tf.int32, tf.float32)
    # when using generator for dataset, it should not contain the batch dim
    output_shapes = (tf.TensorShape((seq_len, 17400)), tf.TensorShape((n_chunks)), tf.TensorShape((n_chunks)), tf.TensorShape((n_chunks)))
    data_set = tf.data.Dataset.from_generator(_g, output_dtypes, output_shapes=output_shapes).map(lambda _a, _b, _c, _d:
                (tf.cast(_a, tf.float32), tf.cast(_b, tf.int32), tf.cast(_c, tf.int32), tf.cast(_d, tf.float32)))
    return data_set


def generate_pure_classification_data_set_from_generator(data_usage=0,intensity=1,im_slice=100, pre_delay=50, post_delay=150,
                                                         pre_chunks=2, resp_chunks=1, post_chunks=1, current_input=True,
                                                         dataset='mnist', path=None, imagenet_img_num=60000, rot90=False,
                                                         from_lgn=True):
    # hard code lgn scale for the case from_lgn=False
    mimc_lgn_std, mimc_lgn_mean = 0.02082, 0.02
    # data_usage: 0, train; 1, test

    if dataset.lower() == 'cifar100':
        all_ds = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    elif dataset.lower() == 'cifar10':
        all_ds = tf.keras.datasets.cifar10.load_data()
    elif dataset.lower() == 'mnist':
        all_ds = tf.keras.datasets.mnist.load_data()
    elif dataset.lower() == 'fashion_mnist':
        all_ds = tf.keras.datasets.fashion_mnist.load_data()

    if data_usage == 0:
        images, labels = all_ds[data_usage]
    else:
        images, labels = all_ds[data_usage]
        # choose fixed validation set to minimize the variance
        # images = images[0:1280] # normally, the batch size is 64
        # labels = labels[0:1280]    

    # LGN module only can receive gray-scale images with the value in [-intensity,intensity] from black to white
    if len(images.shape) > 3:
        images = tf.image.rgb_to_grayscale(images) / 255
    else:
        images = images[...,None]/255

    if rot90:
        images = tf.image.rot90(images)

    lgn = lgn_model.LGN()
    seq_len = pre_delay + im_slice + post_delay
    chunk_size = 50 # ms
    n_chunks = int(seq_len / chunk_size)

    assert n_chunks == resp_chunks + pre_chunks + post_chunks

    def _g():
        for ind in range(images.shape[0]):
            if from_lgn:
                # LGN model only receives 120 x 240, the core part only receives an eclipse TODO
                img = tf.image.resize_with_pad(images[ind], 120, 240, method='lanczos5')
                # maintain the images for a while
                tiled_img = tf.tile(img[None,...], (im_slice, 1, 1, 1))
                # make it in [-intensity, intensity]
                tiled_img = (tiled_img - .5) * intensity / .5
            else:
                # to mimic the 17400 dim of LGN output
                img = tf.image.resize_with_pad(images[ind], 100, 174, method='lanczos5')
                # maintain the images for a while
                tiled_img = tf.tile(img[None,...], (im_slice, 1, 1, 1))

            # add an empty period before a period of real image for continuing classification
            z1 = tf.tile(tf.zeros_like(img)[None,...], (pre_delay, 1, 1, 1))
            z2 = tf.tile(tf.zeros_like(img)[None,...], (post_delay, 1, 1, 1))
            videos = tf.concat((z1, tiled_img, z2), 0)
            if from_lgn:
                spatial = lgn.spatial_response(videos)
                firing_rates = lgn.firing_rates_from_spatial(*spatial)
            else:
                firing_rates = tf.reshape(videos, [-1,17400])
            # sample rate
            # assuming dt = 1 ms
            _p = 1 - tf.exp(-firing_rates / 1000.)
            # _z = tf.cast(fixed_noise < _p, dtype)
            if current_input:
                _z = _p * 1.3
                if not from_lgn:
                    _z = _z * mimc_lgn_std
                    _z = (_z - tf.reduce_mean(_z)) / tf.math.reduce_std(_z) * mimc_lgn_std + mimc_lgn_mean
            else:
                _z = tf.cast(tf.random.uniform(tf.shape(_p)) < _p, tf.float32)
            label = tf.concat([tf.zeros(pre_chunks)] + [labels[ind]*tf.ones(resp_chunks)] + [tf.zeros(post_chunks)],axis=0)
            weight = tf.concat([0*tf.ones(pre_chunks)] + [tf.ones(resp_chunks)] + [0*tf.ones(post_chunks)],axis=0)
            # for plotting, label the image when it holds on
            image_labels = tf.concat([tf.zeros(int(pre_delay/chunk_size))] + [labels[ind]*tf.ones(int(im_slice/chunk_size))] + [tf.zeros(int(post_delay/chunk_size))],axis=0)
            yield _z, label, image_labels, weight

    output_dtypes = (tf.float32, tf.int32, tf.int32, tf.float32)
    # when using generator for dataset, it should not contain the batch dim
    output_shapes = (tf.TensorShape((seq_len, 17400)), tf.TensorShape((n_chunks)), tf.TensorShape((n_chunks)), tf.TensorShape((n_chunks)))
    data_set = tf.data.Dataset.from_generator(_g, output_dtypes, output_shapes=output_shapes).map(lambda _a, _b, _c, _d:
                (tf.cast(_a, tf.float32), tf.cast(_b, tf.int32), tf.cast(_c, tf.int32), tf.cast(_d, tf.float32)))
    return data_set

def make_drifting_grating_stimulus(row_size=120, col_size=240, moving_flag=True, image_duration=100, cpd = 0.05,
            temporal_f = 2, theta = 45, phase = None, contrast = 1.0):
    # parameters from Allen's code
    '''
    Create the grating movie with the desired parameters
    :param t_min: start time in seconds
    :param t_max: end time in seconds
    :param cpd: cycles per degree
    :param temporal_f: in Hz
    :param theta: orientation angle
    :return: Movie object of grating with desired parameters
    '''
    row_size = row_size*2 # somehow, Franz's code only accept larger size; thus, i did the mulitplication
    col_size = col_size*2
    frame_rate = 1000 # Hz
    t_min = 0
    t_max = image_duration/1000
    if phase is None:
        phase = np.random.rand(1)*180

    assert contrast <= 1, "Contrast must be <= 1"
    assert contrast > 0, "Contrast must be > 0"

    physical_spacing = 1. / (float(cpd) * 10)    #To make sure no aliasing occurs
    row_range  = np.linspace(0, row_size, int(row_size / physical_spacing), endpoint = True)
    col_range  = np.linspace(0, col_size, int(col_size / physical_spacing), endpoint = True)
    numberFramesNeeded = int(round(frame_rate * t_max))
    time_range = np.linspace(0, t_max, numberFramesNeeded, endpoint=True)   ### this was a bug... instead of zero it was gray_screen and so time was stretched! Fixed on Jan 11, 2018

    tt, yy, xx = np.meshgrid(time_range, row_range, col_range, indexing='ij')

    thetaRad = np.pi*(180-theta)/180.   #Add negative here to match brain observatory angles!
    phaseRad = np.pi*(180-phase)/180.
    xy = xx * np.cos(thetaRad) + yy * np.sin(thetaRad)
    data = contrast*np.sin(2*np.pi*(cpd * xy + temporal_f *tt) + phaseRad)
    if moving_flag:
        return data.astype(np.float32)
    else:
        return np.tile(data[0].astype(np.float32)[None,...],(image_duration,1,1))

def generate_drifting_grating(orientation=45, intensity=10, im_slice=100, pre_delay=50, post_delay=50,
                                                         current_input=True, from_lgn=True):
    mimc_lgn_std, mimc_lgn_mean = 0.02855, 0.02146

    lgn = lgn_model.LGN()
    seq_len = pre_delay + im_slice + post_delay

    def _g():
        while True:
            if from_lgn:
                tiled_img = make_drifting_grating_stimulus(moving_flag=False, image_duration=im_slice, cpd = 0.05, temporal_f = 2, theta = orientation, phase = None, contrast = 1.0)
                # make it in [-intensity, intensity]
                tiled_img = (tiled_img[...,None] - .5) * intensity / .5
            else:
                tiled_img = make_drifting_grating_stimulus(row_size=100,col_size=174,moving_flag=False, image_duration=im_slice, cpd = 0.05, temporal_f = 2, theta = orientation, phase = None, contrast = 1.0)
                tiled_img = tiled_img[...,None]

            # add an empty period before a period of real image for continuing classification
            z1 = tf.tile(tf.zeros_like(tiled_img[0,...])[None,...], (pre_delay, 1, 1, 1))
            z2 = tf.tile(tf.zeros_like(tiled_img[0,...])[None,...], (post_delay, 1, 1, 1))
            videos = tf.concat((z1, tiled_img, z2), 0)
            if from_lgn:
                spatial = lgn.spatial_response(videos)
                firing_rates = lgn.firing_rates_from_spatial(*spatial)
            else:
                firing_rates = tf.reshape(videos, [-1,17400])
            # sample rate
            # assuming dt = 1 ms
            _p = 1 - tf.exp(-firing_rates / 1000.)
            # _z = tf.cast(fixed_noise < _p, dtype)
            if current_input:
                _z = _p * 1.3
                if not from_lgn:
                    _z = _z * mimc_lgn_std
                    _z = (_z - tf.reduce_mean(_z)) / tf.math.reduce_std(_z) * mimc_lgn_std + mimc_lgn_mean
            else:
                _z = tf.cast(tf.random.uniform(tf.shape(_p)) < _p, tf.float32)

            yield _z

    output_dtypes = (tf.float32)
    # when using generator for dataset, it should not contain the batch dim
    output_shapes = (tf.TensorShape((seq_len, 17400)))
    data_set = tf.data.Dataset.from_generator(_g, output_dtypes, output_shapes=output_shapes).map(lambda _a:
                tf.cast(_a, tf.float32))
    return data_set

def generate_fine_orientation_discrimination(from_lgn=True, intensity=10, im_slice=100, pre_delay=50, post_delay=50,
                                                         pre_chunks=3, resp_chunks=1, post_chunks=0, current_input=True):
    # hard code lgn scale for the case from_lgn=False
    mimc_lgn_std, mimc_lgn_mean = 0.02855, 0.02146

    lgn = lgn_model.LGN()
    seq_len = pre_delay + im_slice + post_delay
    chunk_size = 50 # ms
    n_chunks = int(seq_len / chunk_size)

    assert n_chunks == resp_chunks + pre_chunks + post_chunks

    def _g():
        while True:
            orientation = 45 + tf.math.round(tf.random.uniform(shape=[1],minval=-0.5,maxval=0.5) * 40) / 10 # choose from [43,47] with the precision of 0.1
            if from_lgn:
                tiled_img = make_drifting_grating_stimulus(moving_flag=False, image_duration=im_slice, cpd = 0.05, temporal_f = 2, theta = orientation, phase = None, contrast = 1.0)
                # make it in [-intensity, intensity]
                tiled_img = (tiled_img[...,None] - .5) * intensity / .5
            else:
                tiled_img = make_drifting_grating_stimulus(row_size=100,col_size=174,moving_flag=False, image_duration=im_slice, cpd = 0.05, temporal_f = 2, theta = orientation, phase = None, contrast = 1.0)
                tiled_img = tiled_img[...,None]

            # add an empty period before a period of real image for continuing classification
            z1 = tf.tile(tf.zeros_like(tiled_img[0,...])[None,...], (pre_delay, 1, 1, 1))
            z2 = tf.tile(tf.zeros_like(tiled_img[0,...])[None,...], (post_delay, 1, 1, 1))
            videos = tf.concat((z1, tiled_img, z2), 0)
            if from_lgn:
                spatial = lgn.spatial_response(videos)
                firing_rates = lgn.firing_rates_from_spatial(*spatial)
            else:
                firing_rates = tf.reshape(videos, [-1,17400])
            # sample rate
            # assuming dt = 1 ms
            _p = 1 - tf.exp(-firing_rates / 1000.)
            # _z = tf.cast(fixed_noise < _p, dtype)
            if current_input:
                _z = _p * 1.3
                if not from_lgn:
                    _z = _z * mimc_lgn_std
                    _z = (_z - tf.reduce_mean(_z)) / tf.math.reduce_std(_z) * mimc_lgn_std + mimc_lgn_mean
            else:
                _z = tf.cast(tf.random.uniform(tf.shape(_p)) < _p, tf.float32)
            ground_truth = tf.cast(orientation > 45, tf.float32)
            label = tf.concat([tf.zeros(pre_chunks)] + [ground_truth*tf.ones(resp_chunks)] + [tf.zeros(post_chunks)],axis=0)
            weight = tf.concat([0.0*tf.ones(pre_chunks)] + [tf.ones(resp_chunks)] + [0.0*tf.ones(post_chunks)],axis=0)
            # for plotting, label the image when it holds on
            image_labels = tf.concat([tf.zeros(int(pre_delay/chunk_size))] + [orientation*tf.ones(int(im_slice/chunk_size))] + [tf.zeros(int(post_delay/chunk_size))],axis=0)
            yield _z, label, image_labels, weight

    output_dtypes = (tf.float32, tf.int32, tf.float32, tf.float32)
    # when using generator for dataset, it should not contain the batch dim
    output_shapes = (tf.TensorShape((seq_len, 17400)), tf.TensorShape((n_chunks)), tf.TensorShape((n_chunks)), tf.TensorShape((n_chunks)))
    data_set = tf.data.Dataset.from_generator(_g, output_dtypes, output_shapes=output_shapes).map(lambda _a, _b, _c, _d:
                (tf.cast(_a, tf.float32), tf.cast(_b, tf.int32), tf.cast(_c, tf.float32), tf.cast(_d, tf.float32)))
    return data_set

def generate_VCD_orientation(intensity=2, im_slice=100, pre_delay=50, post_delay=150, p_reappear=0.5, pairs_in_epoch=50,
                                                        batch_size=2, current_input=True):

    lgn = lgn_model.LGN()
    seq_len = (pre_delay + im_slice + post_delay)*2
    chunk_size = 50 # ms
    n_chunks = int(seq_len / chunk_size)
    resp_chunks = 1 # 50 ms response window
    pre_chunks = 4 # include 50 ms predelay, 100 ms image, 50 ms post delay
    post_chunks = 1 # 50 ms delay after response

    assert n_chunks == 2*(resp_chunks + pre_chunks + post_chunks)

    changes = np.random.uniform(size=[batch_size*pairs_in_epoch,2]) > p_reappear
    for i in range(batch_size):
        changes[i*pairs_in_epoch,0] = 0 # the first one cannot change

    orientations = []
    for i, change in enumerate(changes):
        if change[0]:
            new_ori = 135 + np.round(np.random.uniform(low=-0.5,high=0.5) * 300) / 10
            while new_ori == orientations[-1]:
                # choose from [120,150] with the precision of 0.1
                new_ori = 135 + np.round(np.random.uniform(low=-0.5,high=0.5) * 300) / 10
            orientations.append(new_ori)
        else:
            if i < 1:
                orientations.append(135 + np.round(np.random.uniform(low=-0.5,high=0.5) * 300) / 10)
            else:
                orientations.append(orientations[-1])

        if change[1]:
            new_ori = 135 + np.round(np.random.uniform(low=-0.5,high=0.5) * 300) / 10
            while new_ori == orientations[-1]:
                # choose from [120,150] with the precision of 0.1
                new_ori = 135 + np.round(np.random.uniform(low=-0.5,high=0.5) * 300) / 10
            orientations.append(new_ori)
        else:
            orientations.append(orientations[-1])

    orientations = np.array(orientations).reshape(batch_size, -1, 2)
    changes = changes.reshape(batch_size, -1, 2)
    # re-arrange for batches
    temp_ori = []
    temp_cha = []
    for i in range(batch_size):
        temp_ori.append(orientations[i,...])
        temp_cha.append(changes[i,...])

    orientations = np.concatenate(temp_ori, axis=1)
    changes = np.concatenate(temp_cha, axis=1)
    orientations = orientations.reshape(-1,2)
    changes = changes.reshape(-1,2)

    orientations = tf.cast(tf.convert_to_tensor(orientations), tf.float32)
    changes = tf.cast(tf.convert_to_tensor(changes), tf.float32)

    def gen_one_video(orientation):
        tiled_img = make_drifting_grating_stimulus(moving_flag=False, image_duration=im_slice, cpd = 0.05, temporal_f = 2, theta = orientation, phase = None, contrast = 1.0)
        # make it in [-intensity, intensity]
        tiled_img = (tiled_img[...,None] - .5) * intensity / .5
        # add an empty period before a period of real image for continuing classification
        z1 = tf.tile(tf.zeros_like(tiled_img[0,...])[None,...], (pre_delay, 1, 1, 1))
        z2 = tf.tile(tf.zeros_like(tiled_img[0,...])[None,...], (post_delay, 1, 1, 1))
        video = tf.concat((z1, tiled_img, z2), 0)
        return video, orientation

    def _g():
        for change, orientaion in zip(changes, orientations):
            video1, orientation1 = gen_one_video(orientaion[0])
            video2, orientation2 = gen_one_video(orientaion[1])

            videos = tf.concat((video1, video2), 0)
            spatial = lgn.spatial_response(videos)
            firing_rates = lgn.firing_rates_from_spatial(*spatial)
            # sample rate
            # assuming dt = 1 ms
            _p = 1 - tf.exp(-firing_rates / 1000.)
            # _z = tf.cast(fixed_noise < _p, dtype)
            if current_input:
                _z = _p * 1.3
            else:
                _z = tf.cast(tf.random.uniform(tf.shape(_p)) < _p, tf.float32)
            ground_truth = tf.cast(change, tf.float32)
            label = tf.concat([tf.zeros(pre_chunks)] + [ground_truth[0]*tf.ones(resp_chunks)] + [tf.zeros(post_chunks)] +\
                [tf.zeros(pre_chunks)] + [ground_truth[1]*tf.ones(resp_chunks)] + [tf.zeros(post_chunks)],axis=0)
            weight = tf.concat([0.0*tf.ones(pre_chunks)] + [tf.ones(resp_chunks)] + [0.0*tf.ones(post_chunks)] +\
                               [0.0*tf.ones(pre_chunks)] + [tf.ones(resp_chunks)] + [0.0*tf.ones(post_chunks)], axis=0)
            # for plotting, label the image when it holds on
            image_label1 = tf.concat([tf.zeros(int(pre_delay/chunk_size))] + [orientation1*tf.ones(int(im_slice/chunk_size))] + [tf.zeros(int(post_delay/chunk_size))],axis=0)
            image_label2 = tf.concat([tf.zeros(int(pre_delay/chunk_size))] + [orientation2*tf.ones(int(im_slice/chunk_size))] + [tf.zeros(int(post_delay/chunk_size))],axis=0)
            image_labels = tf.concat([image_label1,image_label2],axis=0)
            yield _z, label, image_labels, weight

    output_dtypes = (tf.float32, tf.int32, tf.float32, tf.float32)
    # when using generator for dataset, it should not contain the batch dim
    output_shapes = (tf.TensorShape((seq_len, 17400)), tf.TensorShape((n_chunks)), tf.TensorShape((n_chunks)), tf.TensorShape((n_chunks)))
    data_set = tf.data.Dataset.from_generator(_g, output_dtypes, output_shapes=output_shapes).map(lambda _a, _b, _c, _d:
                (tf.cast(_a, tf.float32), tf.cast(_b, tf.int32), tf.cast(_c, tf.float32), tf.cast(_d, tf.float32)))
    return data_set


def generate_evidence_accumulation(path, batch_size, seq_len=700, pause=200, n_cues=7, cue_len=30, interval_len=40, recall_len=80,
        n_examples_per_epoch=100):
    assert seq_len % 50 == 0
    n_chunks_t = (seq_len - pause) // 50
    n_chunks_p = pause // 50
    with open(path, 'rb') as f:
        firing_rates = pkl.load(f)
    np_resting = firing_rates['resting'].astype(np.float32)
    np_stimuli = np.stack((firing_rates['left'], firing_rates['right']), 0).astype(np.float32)
    np_stimuli = np.tile(np_stimuli[:, None], (1, cue_len, 1))
    np_stimuli = np.concatenate((np_stimuli, np.tile(np_resting[None, None], (2, interval_len - cue_len, 1))), 1)
    np_recall = firing_rates['recall'].astype(np.float32)
    with tf.device('/cpu'):
        resting = tf.Variable(np_resting, trainable=False)
        stimuli = tf.Variable(np_stimuli, trainable=False)
        recall = tf.Variable(np_recall, trainable=False)

    def gen_seq(_):
        stim_id = tf.cast(tf.random.uniform((batch_size, n_cues)) * 2, tf.int32)
        t_label = tf.cast(tf.reduce_mean(tf.cast(stim_id, tf.float32), 1) > .5, tf.int32)
        t_label = tf.concat((
            tf.zeros((batch_size, n_chunks_t - 1), tf.int32),
            t_label[..., None], tf.zeros((batch_size, n_chunks_p), tf.int32)), -1)
        t_w = tf.concat((
            tf.zeros((batch_size, n_chunks_t - 1)), tf.ones((batch_size, 1)),
            tf.zeros((batch_size, n_chunks_p))), -1)
        t_stim = tf.gather(stimuli, stim_id, axis=0)
        t_stim = tf.reshape(t_stim, (batch_size, interval_len * n_cues, -1))
        t_pause = tf.tile(resting[None, None], (batch_size, seq_len - pause - n_cues * interval_len - recall_len, 1))
        t_pause_2 = tf.tile(resting[None, None], (batch_size, pause, 1))
        t_recall = tf.tile(recall[None, None], (batch_size, recall_len, 1))
        t_task = tf.concat((t_stim, t_pause, t_recall, t_pause_2), 1)

        t_task = 1 - tf.exp(-t_task / 1000.)
        return t_task * 1.3, t_label, t_label, t_w

    data_set = tf.data.Dataset.from_tensor_slices([0]).map(gen_seq).repeat(n_examples_per_epoch)
    return data_set

def generate_evidence_accumulation_via_LGN(file_name, seq_len=600, pause=250, n_cues=5, cue_len=50, interval_len=10, recall_len=50, post_chunks=0):

    lgn = lgn_model.LGN()
    assert seq_len % 50 == 0
    assert seq_len == pause + n_cues*(cue_len + interval_len) + recall_len
    n_chunks = seq_len // 50

    f = h5py.File(file_name, 'r')
    left_cue = f['left_cue'][()]
    right_cue = f['right_cue'][()]
    gap = f['gap'][()]
    recall = f['recall'][()]
    f.close()
    left_cue = np.tile(left_cue[None,...,None], [cue_len,1,1,1])
    right_cue = np.tile(right_cue[None,...,None], [cue_len,1,1,1])
    gap_between_cues = np.tile(gap[None,...,None], [interval_len,1,1,1])
    left_cue = np.concatenate((left_cue, gap_between_cues),axis=0)
    right_cue = np.concatenate((right_cue, gap_between_cues),axis=0)
    cues = tf.Variable(tf.stack((left_cue,right_cue),axis=0), trainable=False)

    recall = np.tile(recall[None,...,None], [recall_len,1,1,1])
    delay = np.tile(gap[None,...,None], [pause,1,1,1])


    def gen_seq():
        while True:
            stim_id = tf.cast(tf.random.uniform((n_cues,)) * 2, tf.int32)
            t_label = tf.cast(tf.reduce_mean(tf.cast(stim_id, tf.float32), 0) > .5, tf.int32)
            t_label = tf.concat((tf.zeros(n_chunks - 1, tf.int32), t_label[..., None]), -1)
            t_w = tf.concat((tf.zeros(n_chunks - post_chunks-1,), tf.ones(1), tf.zeros(post_chunks,)), -1)
            t_stim = tf.reshape(tf.gather(cues, stim_id, axis=0), [-1,120,240,1])
            t_task = tf.cast(tf.concat((t_stim, delay, recall), 0),tf.float32)
            spatial = lgn.spatial_response(t_task)
            firing_rates = lgn.firing_rates_from_spatial(*spatial)
            _p = 1.3*(1 - tf.exp(-firing_rates / 1000.))
            image_label = tf.concat([stim_id, tf.cast(tf.zeros(int((recall_len + pause + 50)/50)),tf.int32)],axis=0) # 50 is chunk size; stim_id span on 60*5 ms (not 50 ms chunk) so I conpensate it with an extra dummy zeros
            yield _p, t_label, image_label, t_w

    output_dtypes = (tf.float32, tf.int32, tf.int32, tf.float32)
    output_shapes = (tf.TensorShape((seq_len, 17400)), tf.TensorShape((n_chunks)), tf.TensorShape((n_chunks)), tf.TensorShape((n_chunks)))
    data_set = tf.data.Dataset.from_generator(gen_seq, output_dtypes, output_shapes=output_shapes).map(lambda _a, _b, _c, _d:
                (tf.cast(_a, tf.float32), tf.cast(_b, tf.int32), tf.cast(_c, tf.int32), tf.cast(_d, tf.float32)))
    return data_set
