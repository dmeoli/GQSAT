# Config file according to GraphQSAT paper: https://arxiv.org/pdf/1909.11830.pdf
# Appendix: C.4

from itertools import chain

DQN = {
    "batch_updates": 50000,
    "lr": 2e-4,
    "bsize": 64,
    "buffer_size": 20000,
    "max_cap_fill_buffer": 0,
    "history_len": 1,
    "priority_alpha": 0.5,
    "priority_beta": 0.5,
    "eps_init": 1.0,
    "eps_final": 0.01,
    "eps_decay_steps": 30000,
    "init_exploration_steps": 5000,
    "expert_exploration_prob": 0.0,
    "gamma": 0.99,
    "step_freq": 4,
    "target_update_freq": 10,
    "train_time_max_decisions_allowed": 500,
    "test_time_max_decisions_allowed": 500,
    "penalty_size": 0.1,
}

Optimization = {
    "loss": "mse",
    "opt": "adam",
    "lr_scheduler_gamma": 1,
    "lr_scheduler_frequency": 3000,
    "grad_clip": 1,
    "grad_clip_norm_type": 2,
}

GraphNetwork = {
    "core_steps": 4,
    "e2v_aggregator": sum,
    "n_hidden": 1,
    "hidden_size": 64,
    "decoder_v_out_size": 32,
    "decoder_e_out_size": 1,
    "decoder_g_out_size": 1,
    "encoder_v_out_size": 32,
    "encoder_e_out_size": 32,
    "encoder_g_out_size": 32,
    "core_v_out_size": 64,
    "core_e_out_size": 64,
    "core_g_out_size": 32,
    "activation": "relu",
    "independent_block_layers": 0
}

main = {
    "logdir": "./log",
    "env_name": "sat_v0",
    "train_problems_paths": "../data/uniform-random-3-sat/train/uf50_218",
    "eval_problems_paths": "../data/uniform-random-3-sat/val/uf50_218",
    "eval_freq": 1000,
    "eval_time_limit": 3600,
    "save_freq": 500,
}


def dict_union(*args):
    return dict(chain.from_iterable(d.items() for d in args))


all_config = dict_union(DQN, Optimization, GraphNetwork, main)
