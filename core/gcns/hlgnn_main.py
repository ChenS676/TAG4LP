import copy
import os, sys

from torch_sparse import SparseTensor
from torch_geometric.graphgym import params_count


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import torch
from functools import partial
from graphgps.utility.utils import random_sampling

from torch_geometric import seed_everything
from torch_geometric.graphgym.utils.device import auto_select_device
from graphgps.utility.utils import set_cfg, get_git_repo_root_path, custom_set_run_dir, set_printing, run_loop_settings, \
    create_optimizer, config_device, \
    create_logger

from data_utils.load import load_data_lp, load_graph_lp
from graphgps.utility.utils import save_run_results_to_csv
from graphgps.network.hlgnn import HLGNN, LinkPredictor
from graphgps.train.hlgnn_train import Trainer_HLGNN


def hlgnn_dataset(data, splits):
    edge_index = data.edge_index
    data.num_nodes = data.x.shape[0]
    data.edge_weight = None
    data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    data.max_x = -1
    # Use training + validation edges for inference on test set.
    if cfg.data.use_valedges_as_input:
        val_edge_index = splits['valid']['pos_edge_label_index']
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t
    return data
def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='core/yamls/cora/gcns/hlgnn.yaml',
                        help='The configuration file path.')

    parser.add_argument('--sweep', dest='sweep_file', type=str, required=False,
                        default='core/yamls/cora/gcns/hlgnn.yaml',
                        help='The configuration file path.')
    parser.add_argument('--data', dest='data', type=str, required=True,
                        default='cora',
                        help='data name')
    parser.add_argument('--repeat', type=int, default=5,
                        help='The number of repeated jobs.')
    parser.add_argument('--start_seed', type=int, default=0,
                        help='The number of starting seed.')
    parser.add_argument('--batch_size', dest='bs', type=int, required=False,
                        default=2**15,
                        help='data name')
    parser.add_argument('--device', dest='device', required=True, 
                        help='device id')
    parser.add_argument('--epochs', dest='epoch', type=int, required=True,
                        default=400,
                        help='data name')
    parser.add_argument('--wandb', dest='wandb', required=False, 
                        help='data name')
    parser.add_argument('--mark_done', action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    parser.add_argument('--downsampling', type=float, default=1,
                        help='Downsampling rate.')
    return parser.parse_args()




if __name__ == "__main__":
    FILE_PATH = f'{get_git_repo_root_path()}/'

    args = parse_args()
    cfg = set_cfg(FILE_PATH, args.cfg_file)
    cfg.merge_from_list(args.opts)

    cfg.data.name = args.data

    cfg.data.device = args.device
    cfg.model.device = args.device
    cfg.device = args.device
    cfg.train.epochs = args.epoch

    torch.set_num_threads(cfg.num_threads)
    batch_sizes = [cfg.train.batch_size]

    best_acc = 0
    best_params = {}
    loggers = create_logger(args.repeat)
    cfg.device = args.device



    for run_id in range(args.repeat):
        seed = run_id + args.start_seed
        custom_set_run_dir(cfg, run_id)
        print_logger = set_printing(cfg)
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        cfg = config_device(cfg)
        start = time.time()
        splits, __, data = load_data_lp[cfg.data.name](cfg.data)
        splits = random_sampling(splits, args.downsampling)

        data.edge_index = splits['train']['pos_edge_label_index']
        data = hlgnn_dataset(data, splits).to(cfg.device)
        path = f'{os.path.dirname(__file__)}/hlgnn_{cfg.data.name}'
        dataset = {}

        model = HLGNN(data, cfg.model).to(cfg.device)

        predictor = LinkPredictor(data.num_features, cfg.model.hidden_channels, 1, cfg.model.mlp_num_layers, cfg.model.dropout).to(
            cfg.device)

        para_list = list(model.parameters()) + list(predictor.parameters())
        total_params = sum(p.numel() for param in para_list for p in param)
        total_params_print = f'Total number of model parameters is {total_params}'
        print(total_params_print)
        optimizer = torch.optim.Adam(list(predictor.parameters()) + list(model.parameters()), lr=cfg.optimizer.lr)

        # Execute experiment
        trainer = Trainer_HLGNN(FILE_PATH,
                               cfg,
                               model,
                               predictor,
                               optimizer,
                               data,
                               splits,
                               run_id,
                               args.repeat,
                               loggers,
                               print_logger=print_logger,
                               batch_size=cfg.train.batch_size)

        start = time.time()
        trainer.train()
        end = time.time()
        print('Training time: ', end - start)
        save_run_results_to_csv(cfg, loggers, seed, run_id)


    print('All runs:')

    result_dict = {}
    for key in loggers:
        print(key)
        _, _, _, valid_test, _, _ = trainer.loggers[key].calc_all_stats()
        result_dict[key] = valid_test

    trainer.save_result(result_dict)

    cfg.model.params = params_count(model)
    print_logger.info(f'Num parameters: {cfg.model.params}')
    trainer.finalize()
    print_logger.info(f"Inference time: {trainer.run_result['eval_time']}")
