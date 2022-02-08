###################################################
################ Torch Setting ####################
###################################################
device = 0
print_log_interval = 10

###################################################
################# Train Setting ###################
###################################################
train_times  = int(50e10)
TEST = False
random_den = False
size_var = 0.1

###################################################
############## Environment Setting ################
###################################################
env_name = 'PctDiscrete-v0'
given_problem = False
normFactor = 1
seed = int(4)
num_processes = 64      # batch size
batch_size = num_processes
model_save_interval = 200
model_update_interval = 20e3

internal_node_holder = 80
internal_node_length = 6
leaf_node_holder = 50
next_holder = 1
shuffle = False
graph_size = leaf_node_holder + next_holder
eval_freq = 1000
eval_times = 100

###################################################
################# PointerNet Setting ##############
###################################################
embedding_size = 64             # Embedding size
nof_lstms = 1                    # Number of LSTM layers
hiddens = 128                    # Number of hidden units
learning_rate = 1e-6
value_loss_coef  = 1  # (default: 1)
policy_loss_coef = 1  # (default: 1)
entropy_coef = 0.0     # (default: 0)
max_grad_norm = 0.5
model_save_dir = './logs/experiment'
num_steps = int(5)
load_state_dict = False
PCT_model_path = './checkpoints/ems_models/strictEMS-2021.09.03-11-57-52_2021.09.05-06-53-05.pt'  #
use_acktr = True
load_test_data = False
test_data_name = './data/test_data/125_rs_icra.pt'

###################################################
################# Eval Setting ####################
###################################################

if TEST:
    eval_times = 100
    load_state_dict = True
    env_name = 'Ems-v27'
    device = 0
    internal_node_holder = 80
    leaf_node_holder = 300
    next_holder = 1
    shuffle = True
    graph_size = internal_node_holder + leaf_node_holder + next_holder
    PCT_model_path = './checkpoints/ems_models/upper-ems_300-2021.08.29-09-11-14_2021.08.30-09-14-19.pt'
    num_processes = 1
    load_test_data = True
    test_data_name = './test_data/125_rs_icra.pt'
    batch_size = num_processes
    factor = 1

# from gym.envs.registration import register
# register(
#     id='PctDiscrete-v0',                                              # Format should be xxx-v0, xxx-v1
#     entry_point='pct_envs.PctDiscrete0:PackingDiscrete',              # Expalined in envs/__init__.py
# )
# register(
#     id='PctContinuous-v0',
#     entry_point='pct_envs.PctContinuous0:PackingContinuous',
# )