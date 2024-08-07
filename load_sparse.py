import os
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
import pickle as pkl


def sort_indices(indices, weights):
    max_ind = np.max(indices) + 1
    q = indices[:, 0] * max_ind + indices[:, 1]
    sorted_ind = np.argsort(q)
    return indices[sorted_ind], weights[sorted_ind]


def load_network(path='/data/allen/v1_model/GLIF_network/network_dat.pkl',
                 h5_path='/data/allen/v1_model/GLIF_network/network/v1_nodes.h5',
                 data_dir = '.',
                 core_only=True, n_neurons=None, seed=3000, connected_selection=False,
                 use_rand_ini_w=False, use_dale_law=True, use_rand_connectivity=False,
                 use_uniform_neuron_type=False, use_only_one_type=False, scale_w_e=-1):
    rd = np.random.RandomState(seed=seed)

    with open(path, 'rb') as f:
        d = pkl.load(f)

    n_nodes = sum([len(a['ids']) for a in d['nodes']])
    n_edges = sum([len(a['source']) for a in d['edges']])
    # max_delay = max([a['params']['delay'] for a in d['edges']])

    bmtk_id_to_tf_id = np.arange(n_nodes)
    tf_id_to_bmtk_id = np.arange(n_nodes)

    edges = d['edges']
    h5_file = h5py.File(h5_path, 'r')
    assert np.diff(h5_file['nodes']['v1']['node_id']).var() < 1e-12
    x = np.array(h5_file['nodes']['v1']['0']['x'])
    y = np.array(h5_file['nodes']['v1']['0']['y'])
    z = np.array(h5_file['nodes']['v1']['0']['z'])
    r = np.sqrt(x ** 2 + z ** 2)

    if connected_selection:
        sorted_ind = np.argsort(r)
        sel = np.zeros(n_nodes, np.bool)
        sel[sorted_ind[:n_neurons]] = True
        print(f'> Maximum sample radius: {r[sorted_ind[n_neurons - 1]]:.2f}')
    elif core_only:
        sel = r < 400
        if n_neurons is not None and n_neurons > 0:
            inds, = np.where(sel)
            take_inds = rd.choice(inds, size=n_neurons, replace=False)
            sel[:] = False
            sel[take_inds] = True
    elif n_neurons is not None and n_neurons > 0:
        legit_neurons = np.arange(n_nodes)
        take_inds = rd.choice(legit_neurons, size=n_neurons, replace=False)
        sel = np.zeros(n_nodes, np.bool)
        sel[take_inds] = True
    n_nodes = np.sum(sel)
    tf_id_to_bmtk_id = tf_id_to_bmtk_id[sel]
    bmtk_id_to_tf_id = np.zeros_like(bmtk_id_to_tf_id) - 1
    for tf_id, bmtk_id in enumerate(tf_id_to_bmtk_id):
        bmtk_id_to_tf_id[bmtk_id] = tf_id
    x = x[sel]
    y = y[sel]
    z = z[sel]

    n_edges = 0
    for edge in edges:
        target_tf_ids = bmtk_id_to_tf_id[np.array(edge['target'])]
        source_tf_ids = bmtk_id_to_tf_id[np.array(edge['source'])]
        edge_exists = np.logical_and(target_tf_ids >= 0, source_tf_ids >= 0)
        n_edges += np.sum(edge_exists)

    print(f'> Number of Neurons: {n_nodes}')
    print(f'> Number of Synapses: {n_edges}')

    n_node_types = len(d['nodes'])
    node_params = dict(
        V_th=np.zeros(n_node_types, np.float32),
        g=np.zeros(n_node_types, np.float32),
        E_L=np.zeros(n_node_types, np.float32),
        k=np.zeros((n_node_types, 2), np.float32),
        C_m=np.zeros(n_node_types, np.float32),
        V_reset=np.zeros(n_node_types, np.float32),
        tau_syn=np.zeros((n_node_types, 4), np.float32),
        t_ref=np.zeros(n_node_types, np.float32),
        asc_amps=np.zeros((n_node_types, 2), np.float32)
    )
    node_type_ids = np.zeros(n_nodes, np.int64)
    for i, node_type in enumerate(d['nodes']):
        tf_ids = bmtk_id_to_tf_id[np.array(node_type['ids'])]
        tf_ids = tf_ids[tf_ids >= 0]
        node_type_ids[tf_ids] = i
        for k, v in node_params.items():
            v[i] = node_type['params'][k]

    if use_uniform_neuron_type: # remove diversity of neurons, keep just one GLIF3 model for E and one for I
        node_params = dict(
            V_th=np.zeros(n_node_types, np.float32),
            g=np.zeros(n_node_types, np.float32),
            E_L=np.zeros(n_node_types, np.float32),
            k=np.zeros((n_node_types, 2), np.float32),
            C_m=np.zeros(n_node_types, np.float32),
            V_reset=np.zeros(n_node_types, np.float32),
            tau_syn=np.zeros((n_node_types, 4), np.float32),
            t_ref=np.zeros(n_node_types, np.float32),
            asc_amps=np.zeros((n_node_types, 2), np.float32)
        )

        df = pd.read_csv(os.path.join(data_dir, 'network/v1_node_types.csv'), delimiter=' ')
        # note that the node_type_ids is not changed in this control setting for other code
        for i, node_type in enumerate(d['nodes']):
            tf_ids = bmtk_id_to_tf_id[np.array(node_type['ids'])]
            tf_ids = tf_ids[tf_ids >= 0]
            if use_only_one_type:
                for k, v in node_params.items():
                     v[i] = d['nodes'][19]['params'][k]# e23Cux2
            else:
                if df.iloc[i]['pop_name'].startswith('e'):
                    for k, v in node_params.items():
                        v[i] = d['nodes'][18]['params'][k]# e23Cux2
                elif df.iloc[i]['pop_name'].startswith('i'):
                    for k, v in node_params.items():
                        v[i] = d['nodes'][23]['params'][k]#i23Pvalb
                else:
                    raise ValueError('It is neither excitatory nor inhibitory; something is wrong with your file!')


    dense_shape = (4 * n_nodes, n_nodes)
    indices = np.zeros((n_edges, 2), dtype=np.int64)
    weights = np.zeros(n_edges, np.float32)
    delays = np.zeros(n_edges, np.float32)

    current_edge = 0
    for edge in edges:
        r = edge['params']['receptor_type'] - 1
        target_tf_ids = bmtk_id_to_tf_id[np.array(edge['target'])]
        source_tf_ids = bmtk_id_to_tf_id[np.array(edge['source'])]
        edge_exists = np.logical_and(target_tf_ids >= 0, source_tf_ids >= 0)
        target_tf_ids = target_tf_ids[edge_exists]
        source_tf_ids = source_tf_ids[edge_exists]
        weights_tf = edge['params']['weight'][edge_exists]
        delays_tf = edge['params']['delay']
        n_new_edge = np.sum(edge_exists)
        indices[current_edge:current_edge + n_new_edge] = np.array([target_tf_ids * 4 + r, source_tf_ids]).T
        weights[current_edge:current_edge + n_new_edge] = weights_tf
        delays[current_edge:current_edge + n_new_edge] = delays_tf
        current_edge += n_new_edge

    indices, weights = sort_indices(indices, weights)

    if use_rand_connectivity: # break laminar structure and other structured connectivity; keeping the number of connectivities; Note that it is not initial condition but a sustained bias
        indices = np.zeros((n_edges, 2), dtype=np.int64)
        # tmp = rd.choice(int(dense_shape[0] * dense_shape[1]), weights.size, replace=False)
        # post_indices= np.mod(tmp, dense_shape[0])
        # pre_indices = tmp//dense_shape[0]
        # indices = np.stack([post_indices, pre_indices], -1)
        with open(os.path.join(data_dir, '../random_connectivity.pkl'),'rb') as f:
            data_tmp = pkl.load(f)
        indices = data_tmp['indices']
        indices, weights = sort_indices(indices, weights)

    if use_rand_ini_w:
        # make the random weights have the same mean and std with the original ones; maintain the E I split as well
        if use_dale_law:
            w_ab_value = np.abs(rd.randn(*weights.shape))
            w_ab_value = (w_ab_value - w_ab_value.mean()) / w_ab_value.std()
            w_ab_value = w_ab_value*weights.std() + weights.mean()
            rand_w = np.sign(weights)*w_ab_value # Dale's law
        else:
            rand_w = rd.randn(*weights.shape)*weights.std() + weights.mean()
        weights = rand_w.astype('float32')

    if scale_w_e > 0:
        weights[weights > 0] = weights[weights > 0] * scale_w_e

    network = dict(
        x=x, y=y, z=z,
        n_nodes=n_nodes,
        n_edges=n_edges,
        node_params=node_params,
        node_type_ids=node_type_ids,
        synapses=dict(indices=indices, weights=weights, delays=delays, dense_shape=dense_shape),
        tf_id_to_bmtk_id=tf_id_to_bmtk_id,
        bmtk_id_to_tf_id=bmtk_id_to_tf_id
    )

    return network


def load_input(path='/data/allen/v1_model/GLIF_network/input_dat.pkl',
               start=0,
               duration=3000,
               dt=1,
               bmtk_id_to_tf_id=None):
    with open(path, 'rb') as f:
        d = pkl.load(f)

    input_populations = []
    for input_population, input_pop_ind in zip(d,range(len(d))):
        post_indices = []
        pre_indices = []
        weights = []
        delays = []

        for edge in input_population[1]:
            r = edge['params']['receptor_type'] - 1
            target_tf_id = np.array(edge['target'])
            source_tf_id = np.array(edge['source'])
            weights_tf = np.array(edge['params']['weight'])
            delays_tf = np.zeros_like(weights_tf) + edge['params']['delay']
            if bmtk_id_to_tf_id is not None:
                target_tf_id = bmtk_id_to_tf_id[target_tf_id]
                edge_exists = target_tf_id >= 0
                target_tf_id = target_tf_id[edge_exists]
                source_tf_id = source_tf_id[edge_exists]
                weights_tf = weights_tf[edge_exists]
                delays_tf = delays_tf[edge_exists]
            post_indices.extend(4 * target_tf_id + r)
            pre_indices.extend(source_tf_id)
            weights.extend(weights_tf)
            delays.append(delays_tf)
        indices = np.stack([post_indices, pre_indices], -1)
        weights = np.array(weights)
        delays = np.concatenate(delays)
        indices, weights = sort_indices(indices, weights)

        n_neurons = len(input_population[0]['ids'])
        spikes = np.zeros((int(duration / dt), n_neurons))
        for i, sp in zip(input_population[0]['ids'], input_population[0]['spikes']):
            sp = sp[np.logical_and(start < sp, sp < start + duration)] - start
            sp = (sp / dt).astype(np.int)
            for s in sp:
                spikes[s, i] += 1

        input_populations.append(dict(
            n_inputs=n_neurons, indices=indices.astype(np.int64), weights=weights, delays=delays, spikes=spikes))
    return input_populations

def load_TD_input(path, network, n_inputs, targets, inter_area_min_delay, inter_area_max_delay, seed):
    with open(path, 'rb') as f:
        d = pkl.load(f)

    # get the connection probability of bottom-up
    cons = [edge['target'].__len__() for edge in d[0][1]] # d[0] stimulus, d[1] background
    num_cons = sum(cons)
    con_prob = num_cons / (17400 * (network['laminar_indices']['L4e'].size + network['laminar_indices']['L4i'].size))
    # get the pool of connection strength and resample it for top-down connections
    weights = [edge['params']['weight'] for edge in d[0][1]]
    weights_pool = np.concatenate(weights)

    # parse the target populations
    targets_pool = []
    for target_pop in targets.split(','):
        targets_pool.append(network['laminar_indices'][target_pop])

    post_indices = []
    pre_indices = []
    weights = []
    rd = np.random.RandomState(seed=seed)
    for target in targets_pool:
        tmp = rd.choice(int(target.size * n_inputs), int(0.1*con_prob * target.size * n_inputs), replace=False)
        post_indices.extend(target[np.mod(tmp, target.size)]*4 + rd.randint(0,4,tmp.size))
        pre_indices.extend(tmp//target.size)
        weights.extend(rd.choice(weights_pool, tmp.size, replace=True))
    indices = np.stack([post_indices, pre_indices], -1)
    weights = np.array(weights)
    indices, weights = sort_indices(indices, weights)
    delays = rd.randint(low=inter_area_min_delay, high=inter_area_max_delay, size=weights.shape)

    input_populations = dict(
            n_inputs=n_inputs, indices=indices.astype(np.int64), weights=weights, delays=delays)
    return input_populations

def reduce_input_population(input_population, new_n_input, seed=3000):
    rd = np.random.RandomState(seed=seed)

    in_ind = input_population['indices']
    in_weights = input_population['weights']

    assignment = rd.choice(np.arange(new_n_input), size=input_population['n_inputs'], replace=True)
    weight_dict = dict()
    for input_neuron in range(input_population['n_inputs']):
        assigned_neuron = assignment[input_neuron]
        sel = in_ind[:, 1] == input_neuron
        sel_post_inds = in_ind[sel, 0]
        sel_weights = in_weights[sel]
        for post_ind, weight in zip(sel_post_inds, sel_weights):
            t_inds = post_ind, assigned_neuron
            if t_inds not in weight_dict.keys():
                weight_dict[t_inds] = 0.
            weight_dict[t_inds] += weight
    n_synapses = len(weight_dict)
    new_in_ind = np.zeros((n_synapses, 2), np.int64)
    new_in_weights = np.zeros(n_synapses)
    for i, (t_ind, w) in enumerate(weight_dict.items()):
        new_in_ind[i] = t_ind
        new_in_weights[i] = w
    new_in_ind, new_in_weights = sort_indices(new_in_ind, new_in_weights)
    new_input_population = dict(n_inputs=new_n_input, indices=new_in_ind, weights=new_in_weights, spikes=None)
    return new_input_population

def set_laminar_indices(df, h5_path, network, L2_neuron_ratio=0.5):
    # locate neuron population
    node_types = df
    node_h5 = h5py.File(h5_path, mode='r')
    node_type_id_to_pop_name = dict()
    for nid in np.unique(node_h5['nodes']['v1']['node_type_id']):
        ind_list = np.where(node_types.node_type_id == nid)[0]
        assert len(ind_list) == 1
        node_type_id_to_pop_name[nid] = node_types.pop_name[ind_list[0]]

    all_pop_names = []
    for nid in node_h5['nodes']['v1']['node_type_id']:
        all_pop_names.append(node_type_id_to_pop_name[nid])
    all_pop_names = np.array(all_pop_names)[network['tf_id_to_bmtk_id']]

    neuron_pop_id_to_name = ['i1Htr3a', 'e23', 'i23Pvalb', 'i23Sst', 'i23Htr3a', 'e4', 'i4Pvalb', 'i4Sst', 'i4Htr3a', 'e5', 'i5Pvalb', 'i5Sst', 'i5Htr3a', 'e6', 'i6Pvalb', 'i6Sst', 'i6Htr3a']
    neuron_pop_name_to_id = dict()
    for i, name in enumerate(neuron_pop_id_to_name):
        neuron_pop_name_to_id[name] = i

    rough_neuron_pop_names = np.zeros_like(all_pop_names, np.int32)
    for i, pop_name in enumerate(all_pop_names):
        for j, pp_name in enumerate(neuron_pop_id_to_name):
            if pop_name.startswith(pp_name):
                rough_neuron_pop_names[i] = j
                break
    network['laminar_indices'] = dict()
    # exc neurons
    network['laminar_indices'][f'L{1}e'] = np.array([])
    exc_ind = [1,5,9,13]
    for i, layer_number in enumerate([23,4,5,6]):
        network['laminar_indices'][f'L{layer_number}e'] = np.where(rough_neuron_pop_names==exc_ind[i])[0]

    # exc neurons
    network['laminar_indices'][f'L{1}i'] = np.where(rough_neuron_pop_names==0)[0]
    inh_ind = [2,6,10,14]
    for i, layer_number in enumerate([23,4,5,6]):
        temp = []
        for ii in range(3):
            temp.append(np.where(rough_neuron_pop_names==inh_ind[i]+ii)[0])
        network['laminar_indices'][f'L{layer_number}i'] = np.concatenate(temp)   

        
    # split 2 3 layers
    vertical_coordinates_e = network['y'][network['laminar_indices']['L23e']]
    vertical_coordinates_i = network['y'][network['laminar_indices']['L23i']]
    vertical_coordinates = np.hstack((vertical_coordinates_e,vertical_coordinates_i))
    L23_argindices_sorted = np.argsort(vertical_coordinates)
    L23_neuorn_indices = np.hstack((network['laminar_indices']['L23e'],network['laminar_indices']['L23i']))

    L2_argindices = L23_argindices_sorted[:np.int64(L2_neuron_ratio*vertical_coordinates.size)]
    L2e_argindices = L2_argindices[L2_argindices<vertical_coordinates_e.size]
    network['laminar_indices']['L2e'] = L23_neuorn_indices[L2e_argindices]
    L2i_argindices = L2_argindices[L2_argindices>vertical_coordinates_e.size]
    network['laminar_indices']['L2i'] = L23_neuorn_indices[L2i_argindices]

    L3_argindices = L23_argindices_sorted[np.int64(L2_neuron_ratio*vertical_coordinates.size):]
    L3e_argindices = L3_argindices[L3_argindices<vertical_coordinates_e.size]
    network['laminar_indices']['L3e'] = L23_neuorn_indices[L3e_argindices]
    L3i_argindices = L3_argindices[L3_argindices>vertical_coordinates_e.size]
    network['laminar_indices']['L3i'] = L23_neuorn_indices[L3i_argindices]

    return network


def load_billeh(n_input, n_neurons, core_only, data_dir, seed=3000, connected_selection=False, n_output=2,
                neurons_per_output=16, use_rand_ini_w=False, use_dale_law=True, use_rand_connectivity=False,
                use_uniform_neuron_type=False, use_only_one_type=False, scale_w_e=-1, localized_readout=True,
                TD_input=False, n_TD_input=None, targets=None):
    h5_path = os.path.join(data_dir, 'network/v1_nodes.h5')
    network = load_network(
        path=os.path.join(data_dir, 'network_dat.pkl'),
        h5_path=h5_path, data_dir=data_dir, core_only=core_only, n_neurons=n_neurons, seed=seed,
        connected_selection=connected_selection, use_rand_ini_w=use_rand_ini_w, use_dale_law=use_dale_law,
        use_rand_connectivity=use_rand_connectivity, use_only_one_type=use_only_one_type,
        use_uniform_neuron_type=use_uniform_neuron_type, scale_w_e=scale_w_e)
    inputs = load_input(
        start=1000, duration=1000, dt=1, path=os.path.join(data_dir, 'input_dat.pkl'),
        bmtk_id_to_tf_id=network['bmtk_id_to_tf_id'])

    df = pd.read_csv(os.path.join(data_dir, 'network/v1_node_types.csv'), delimiter=' ')
    network = set_laminar_indices(df, h5_path, network)

    l5e_types_indices = []
    for a in df.iterrows():
        if a[1]['pop_name'].startswith('e5'):
            l5e_types_indices.append(a[0])
    l5e_types_indices = np.array(l5e_types_indices)
    l5e_neuron_sel = np.zeros(network['n_nodes'], np.bool)
    for l5e_type_index in l5e_types_indices:
        is_l5_type = network['node_type_ids'] == l5e_type_index
        l5e_neuron_sel = np.logical_or(l5e_neuron_sel, is_l5_type)
    network['l5e_types'] = l5e_types_indices
    network['l5e_neuron_sel'] = l5e_neuron_sel
    print(f'> Number of L5e Neurons: {np.sum(l5e_neuron_sel)}')    

    # Determine localized readout neurons
    node_h5 = h5py.File(h5_path, mode='r')
    node_type_id_to_pop_name = dict()
    for nid in np.unique(node_h5['nodes']['v1']['node_type_id']):
        ind_list = np.where(df.node_type_id == nid)[0]
        assert len(ind_list) == 1
        node_type_id_to_pop_name[nid] = df.pop_name[ind_list[0]]

    node_type_ids = np.array(node_h5['nodes']['v1']['node_type_id'])

    all_pop_names = []
    for nid in node_h5['nodes']['v1']['node_type_id']:
        all_pop_names.append(node_type_id_to_pop_name[nid])
    all_pop_names = np.array(all_pop_names)[network['tf_id_to_bmtk_id']]

    rough_neuron_pop_names2 = np.zeros_like(all_pop_names, np.int32)
    for i, pop_name in enumerate(all_pop_names):
        if pop_name[0] == 'e':
            rough_neuron_pop_names2[i] = 0
        elif pop_name.count('Htr') > 0:
            rough_neuron_pop_names2[i] = 1
        elif pop_name.count('Sst') > 0:
            rough_neuron_pop_names2[i] = 2
        elif pop_name.count('Pvalb') > 0:
            rough_neuron_pop_names2[i] = 3

    layer_pop_names = np.zeros_like(all_pop_names, np.int32)
    for i, pop_name in enumerate(all_pop_names):
        if pop_name[1] == '1':
            layer_pop_names[i] = 0
        elif pop_name[1] == '2':
            layer_pop_names[i] = 1
        elif pop_name[1] == '4':
            layer_pop_names[i] = 2
        elif pop_name[1] == '5':
            layer_pop_names[i] = 3
        elif pop_name[1] == '6':
            layer_pop_names[i] = 4

    x = network['x']
    y = network['y']
    z = network['z']

    bounds = []
    for i in range(5):
        sel = layer_pop_names == i
        bounds.append((np.min(y[sel]), np.max(y[sel])))

    pos = np.stack((x, z, y), -1)
    origin = np.array([[100, -50, np.array(bounds[3]).mean()]])
    origins = np.tile(np.array([[90, -95, np.array(bounds[3]).mean()]])[None], (15, 1, 1))
    origins[:15, 0, :2] = [
        [0, 0],
        [100, -110],
        [-100, -110],
        [-100, 110],
        [100, 110],
        [0, 260],
        [180, 230],
        [-180, 230],
        [270, 95],
        [-270, 95],
        [270, -95],
        [-270, -95],
        [180, -230],
        [-180, -230],
        [0, -260]
    ]

    if localized_readout:
        try:
            for i in range(15):
                origin = origins[i]
                sel = rough_neuron_pop_names2 == 0
                sel = np.logical_and(sel, y < bounds[3][1])
                sel = np.logical_and(sel, y > bounds[3][0])
                sel = np.logical_and(sel, np.sqrt(np.square(pos - origin).sum(-1)) < 55)
                rd = np.random.RandomState(seed=seed)
                sel_ind = np.where(sel)[0]
                sel_ind = rd.choice(sel_ind, replace=False, size=neurons_per_output)
                sel = np.zeros_like(sel)
                sel[sel_ind] = True
                network[f'localized_readout_neuron_ids_{i}'] = np.where(sel)[0][None]
        except:
            print('Warning: Small neuronal volume, not all readout populations available')
            if 'localized_readout_neuron_ids_0' not in network.keys():
                raise ValueError('Neuronal volume too small: No readout population')
    else:
        readout_neurons_random = rd.choice(l5e_neuron_indices, size=30*15, replace=False)
        readout_neurons_random = readout_neurons_random.reshape((15, 30))
        # I still use localized name but it is not anymore!!!
        for i in range(15):
            network[f'localized_readout_neuron_ids_{i}'] = readout_neurons_random[i,:][None,:]


    network['localized_readout_neuron_ids'] = network['localized_readout_neuron_ids_0']
    # ---------------------------

    input_population = inputs[0]
    bkg = inputs[1]
    bkg_weights = np.zeros((network['n_nodes'] * 4,), np.float32)
    bkg_weights[bkg['indices'][:, 0]] = bkg['weights']
    if n_input != 17400:
        input_population = reduce_input_population(input_population, n_input, seed=seed)

    if TD_input:
        TD_inputs = load_TD_input(os.path.join(data_dir, 'input_dat.pkl'), network, n_TD_input, targets, 3, 5, seed)
        return TD_inputs, input_population, network, bkg_weights
    else:
        return input_population, network, bkg_weights


def cached_load_billeh(n_input, n_neurons, core_only, data_dir, seed=3000, connected_selection=False, n_output=2,
                       neurons_per_output=16, use_rand_ini_w=False, scale_w_e=-1):
    store = False
    input_population, network, bkg_weights = None, None, None
    flag_str = f'in{n_input}_rec{n_neurons}_s{seed}_c{core_only}_con{connected_selection}'
    flag_str += f'_out{n_output}_nper{neurons_per_output}'
    cache_path = f'.cache/billeh_network_{flag_str}.pkl'
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                input_population, network, bkg_weights = pkl.load(f)
                print(f'> Sucessfully restored Billeh model from {cache_path}')
        except Exception as e:
            print(e)
            store = True
    else:
        store = True
    if input_population is None or network is None or bkg_weights is None:
        input_population, network, bkg_weights = load_billeh(
            n_input, n_neurons, core_only, data_dir, seed,
            connected_selection=connected_selection, n_output=n_output,
            neurons_per_output=neurons_per_output, use_rand_ini_w=use_rand_ini_w,
            scale_w_e=scale_w_e, output_pop='readout_neuron_ids')
    if store:
        os.makedirs('.cache', exist_ok=True)
        with open(cache_path, 'wb') as f:
            pkl.dump((input_population, network, bkg_weights), f)
        print(f'> Cached Billeh model in {cache_path}')
    return input_population, network, bkg_weights


def main(base_path):
    TD_input_population, input_population, network, bkg_weights = load_billeh(n_input=17400, n_neurons=5000, core_only=False, data_dir=base_path, seed=3000,
                 connected_selection=True, n_output=2, neurons_per_output=16, use_rand_ini_w=False, use_rand_connectivity=False,
                 use_uniform_neuron_type=False, scale_w_e=-1, TD_input=True, n_TD_input=5000, targets='L23e,L5e')

    TD_input_weights = TD_input_population['weights'].astype(np.float32)
    TD_input_indices = TD_input_population['indices']
    TD_input_dense_shape = (4 * 51978, TD_input_population['n_inputs'])
    sparse_w_in = tf.sparse.SparseTensor(
            TD_input_indices, TD_input_weights, TD_input_dense_shape)
    tf.sparse.to_dense(sparse_w_in)



if __name__ == '__main__':
    import argparse
    import socket
    hostname = socket.gethostname()
    if hostname.count('scherr-pc') > 0:
        _data_dir = '/data/allen/v1_model/GLIF_network'
    elif hostname.count('nvcluster'):
        _data_dir = os.path.expanduser('~/allen/mv1_network/GLIF_network')
    elif hostname.count('pCluster') > 0:
        _data_dir = '/home/guozhang/tf_billeh_column/GLIF_network'
    else:
        _data_dir = ''

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=_data_dir)
    args = parser.parse_args()
    main(args.data_dir)
