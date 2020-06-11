import json
import os
import random
import dgl
import torch
import time
import networkx as nx
import numpy as np

from torch.optim import SGD
from deepgg.dataset import ConstructionSequenceDataset, graph_to_construction_sequence, \
    generate_ws_model_construction_sequence, generate_ba_model_construction_sequence
from deepgg.deepgg import DeepGG
from tqdm import tqdm, trange

"""
    Configuration
"""
experiment_prefix = 'deepgg-'
data_base_path = 'data/'
dataset_prefix = 'ds-'
model_prefix = 'model-'
metadata_prefix = 'metadata-'
generate_prefix = 'generated-'

do_hyperparam_selection = False
param_train_epochs = 3  # 8  15
param_learning_rate = 0.0001
param_deepgg_prop_rounds = 2
param_deepgg_v_max = 150
param_deepgg_node_hidden_size = 16
param_deepgg_generate_size = 200
param_deepgg_generate_v_min = 0
hp_deepgg_generate_v_min_range = (0, 40, 10)
param_dataset_size = 35  # 1000 # number of graphs for training
hp_dataset_size_range = (500, 2001, 500)
param_model_num_v = 25  # 100  # number of vertices for used model
param_model_num_v_min = param_model_num_v
param_model_num_v_max = param_model_num_v
hp_model_num_v_min_range = (20, 50, 5)
hp_model_num_v_max_range = (50, 151, 10)
param_model_er_p = 0.2
hp_model_er_p_range = (0.1, 0.9, 0.2)
param_model_ws_k = 10
hp_model_ws_k_range = (3, 12, 1)
param_model_ws_p = 0.2
hp_model_ws_p_range = (0.1, 0.9, 0.2)
param_model_ba_m = 3
hp_model_ba_m_range = (2, 9, 1)
#device = torch.device('cuda:{d_idx}'.format(d_idx=torch.randint(torch.cuda.device_count(), (1,)).item()) if torch.cuda.is_available() else 'cpu')
device = 'cpu'


def generate_seqs_erdos_renyi_bfs(param_dataset_size: int, param_model_num_v: int, param_model_er_p: float, *args, **kwargs):
    seqs = []
    for _ in range(param_dataset_size):
        seqs.append(graph_to_construction_sequence(nx.erdos_renyi_graph(param_model_num_v, param_model_er_p), traversal='bfs'))
    return seqs


def generate_seqs_erdos_renyi_dfs(param_dataset_size: int, param_model_num_v: int, param_model_er_p: float, *args, **kwargs):
    seqs = []
    for _ in range(param_dataset_size):
        seqs.append(graph_to_construction_sequence(nx.erdos_renyi_graph(param_model_num_v, param_model_er_p), traversal='dfs'))
    return seqs


def generate_seqs_watts_strogatz_process(param_dataset_size: int, param_model_num_v: int, param_model_ws_k: int, param_model_ws_p: float, *args, **kwargs):
    seqs = []
    for _ in range(param_dataset_size):
        seqs.append(generate_ws_model_construction_sequence(param_model_num_v, param_model_ws_k, param_model_ws_p))
    return seqs


def generate_seqs_watts_strogatz_bfs(param_dataset_size: int, param_model_num_v: int, param_model_ws_k: int, param_model_ws_p: float, *args, **kwargs):
    seqs = []
    for _ in range(param_dataset_size):
        seqs.append(graph_to_construction_sequence(nx.watts_strogatz_graph(param_model_num_v, param_model_ws_k, param_model_ws_p), traversal='bfs'))
    return seqs


def generate_seqs_watts_strogatz_dfs(param_dataset_size: int, param_model_num_v: int, param_model_ws_k: int, param_model_ws_p: float, *args, **kwargs):
    seqs = []
    for _ in range(param_dataset_size):
        seqs.append(graph_to_construction_sequence(nx.watts_strogatz_graph(param_model_num_v, param_model_ws_k, param_model_ws_p), traversal='dfs'))
    return seqs


def generate_seqs_barabasi_albert_process(param_dataset_size: int, param_model_num_v: int, param_model_ba_m: int, *args, **kwargs):
    seqs = []
    for _ in range(param_dataset_size):
        seqs.append(generate_ba_model_construction_sequence(param_model_num_v, param_model_ba_m))
    return seqs


def generate_seqs_barabasi_albert_bfs(param_dataset_size: int, param_model_num_v: int, param_model_ba_m: int, *args, **kwargs):
    seqs = []
    for _ in range(param_dataset_size):
        seqs.append(graph_to_construction_sequence(nx.barabasi_albert_graph(param_model_num_v, param_model_ba_m), traversal='bfs'))
    return seqs


def generate_seqs_barabasi_albert_dfs(param_dataset_size: int, param_model_num_v: int, param_model_ba_m: int, *args, **kwargs):
    seqs = []
    for _ in range(param_dataset_size):
        seqs.append(graph_to_construction_sequence(nx.barabasi_albert_graph(param_model_num_v, param_model_ba_m), traversal='dfs'))
    return seqs


generators = {name: globals()[name] for name in globals() if name.startswith('generate_seqs_')}


"""
    Hyperparameter assembly
"""
if do_hyperparam_selection:
    print('Doing hyperparameter selection')
    param_vars = [var_name for var_name in globals() if var_name.startswith('param_')]
    for var_name in param_vars:
        var_range_name = var_name.replace('param_', 'hp_')+'_range'
        if var_range_name in globals():
            choices = np.arange(*globals()[var_range_name])
            the_choice = random.choice(choices)
            the_choice = int(the_choice) if type(the_choice) == np.int64 else float(the_choice) if type(the_choice) in [np.float32, np.float64] else the_choice
            globals()[var_name] = the_choice
            print('Hyperparam choice: {var} = "{value}"'.format(var=var_name, value=the_choice))

"""
    Parameter Assembly
"""
start_time = time.time()
param_selected_model = random.choice(list(generators.keys()))

dataset_file_name = '{exp}t{start}-{ds_prefix}{ds_size}-{model}.json'.format(exp=experiment_prefix, ds_prefix=dataset_prefix, ds_size=param_dataset_size, model=param_selected_model, start=int(start_time))
dataset_file_path = os.path.join(data_base_path, dataset_file_name)
print('Dataset file is %s' % dataset_file_path)

model_file_name = '{exp}t{start}-{prefix}{model}.pth'.format(exp=experiment_prefix, prefix=model_prefix, model=param_selected_model, start=int(start_time))
model_file_path = os.path.join(data_base_path, model_file_name)
print('Model file is %s' % model_file_path)

meta_file_name = '{exp}t{start}-{prefix}{model}.json'.format(exp=experiment_prefix, prefix=metadata_prefix, model=param_selected_model, start=int(start_time))
meta_file_path = os.path.join(data_base_path, meta_file_name)
print('Meta file is %s' % meta_file_path)

generated_file_name = '{exp}t{start}-{prefix}{model}.json'.format(exp=experiment_prefix, prefix=generate_prefix, model=param_selected_model, start=int(start_time))
generated_file_path = os.path.join(data_base_path, generated_file_name)
print('Generated file is %s' % generated_file_path)

# Collect device object also as str info for metadata file
param_device = str(device)

"""
    Save-checks
"""


def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)


if not os.path.exists(data_base_path) or not os.path.isdir(data_base_path):
    os.makedirs(data_base_path)

if os.path.exists(model_file_path):
    print('The model path <{path}> already exists!'.format(path=model_file_path))
    exit(-1)

if os.path.exists(meta_file_path):
    print('The meta file path <{path}> already exists!'.format(path=meta_file_path))
    exit(-1)

if os.path.exists(generated_file_path):
    print('The path for generated sequences <{path}> already exists!'.format(path=generated_file_path))
    exit(-1)

touch(model_file_path)
touch(meta_file_path)
touch(generated_file_path)

"""
    Writing meta data information
"""
params = {name: globals()[name] for name in globals() if name.startswith('param_')}
hyperparams = {name: globals()[name] for name in globals() if name.startswith('hp_')}
metadata = {
    'start_time': start_time,
    'params': params,
    'hyperparams': hyperparams,
    'files': { var: globals()[var] for var in globals() if var.endswith('_file_name') or var.endswith('_file_path')}
}
with open(meta_file_path, 'a') as metadata_handle:
    json.dump(metadata, metadata_handle, indent=1)
print('Written metadata.')
print('Params are:')
print(json.dumps(params, indent=1))

"""
    Creating dataset
"""
if not os.path.exists(dataset_file_path):
    print('Generating dataset of size {size} with {seq_size} per graph.'.format(size=param_dataset_size, seq_size=param_model_num_v))
    sequences = generators[param_selected_model](**params)
    dataset = {
        'creation_time': time.time(),
        'model': param_selected_model,
        'size': param_dataset_size,
        'construction_sequences': sequences
    }
    with open(dataset_file_path, 'a') as handle:
        json.dump(dataset, handle, indent=1)
    print('Written dataset to disk.')

print('Reading dataset into memory.')
with open(dataset_file_path, 'r') as handle:
    dataset_meta = json.load(handle)
dataset = ConstructionSequenceDataset(dataset_meta['construction_sequences'], device=device)
print('Dataset has size {len}.'.format(len=len(dataset)))

"""
    Instantiating new model
"""
model = DeepGG(v_max=param_deepgg_v_max, node_hidden_size=param_deepgg_node_hidden_size, num_prop_rounds=param_deepgg_prop_rounds)
optimizer = SGD(model.parameters(), lr=param_learning_rate)
model.to(device)
model.train()

"""
    Training loop
"""
print('Starting training.', flush=True)
time.sleep(0.1)

ts_losses = []
with tqdm(total=len(dataset)*param_train_epochs) as pbar:
    for cur_epoch in range(param_train_epochs):
        for seq_idx, seq in enumerate(dataset):
            #train_seq = torch.tensor(seq, device=device)
            train_seq = seq

            optimizer.zero_grad()

            empty_graph = dgl.DGLGraph()
            empty_graph.to(device)
            log_loss = model.forward(empty_graph, train_seq)
            loss = log_loss
            loss_value = loss.detach().cpu().numpy()

            # Collect time series information for later analysis
            ts_losses.append(loss_value)

            # Do the optimization step
            loss.backward()
            optimizer.step()

            # Update progress bar after each graph processing
            pbar.set_description('Graph <{g_idx}> Epoch {epoch}/{total_epochs}'.format(g_idx=seq_idx, epoch=cur_epoch+1, total_epochs=param_train_epochs))
            pbar.set_postfix(loss=loss_value)
            pbar.update()

print('Done training')

print('Saving ..')
torch.save(model.state_dict(), model_file_path)
print('Saved.')

"""
    Generating sequences
"""
model.v_min = param_deepgg_generate_v_min
model.eval()
print('Initialized. Generating ..', flush=True)
print('', flush=True)

sequences = []
for _ in trange(param_deepgg_generate_size):
    empty_graph = dgl.DGLGraph()
    empty_graph.to(device)
    graph, seq = model.forward(empty_graph)
    sequences.append(seq)

generated_dataset = {
    'creation_time': time.time(),
    'model': dataset_file_name,
    'size': param_deepgg_generate_size,
    'construction_sequences': sequences
}
with open(generated_file_path, 'a') as handle:
    json.dump(generated_dataset, handle, indent=1)
