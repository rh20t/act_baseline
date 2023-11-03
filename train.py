import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from dataset.rh20t import RH20TDataset as RH20TPretrain, collate_fn
from dataset.utils import compute_dict_mean, set_seed, detach_dict
from models.policy import ACTPolicy

def main(args):
    set_seed(1)
    # command line parameters
    ckpt_dir = args['ckpt_dir']
    task_name = args['task_name']
    dataset_root = args['dataset_root']
    batch_size = args['batch_size']
    num_epochs = args['num_epochs']
    chunk_size = args['chunk_size']
    save_epoch = args['save_epoch']
    resume_ckpt = args['resume_ckpt']

    # get task parameters
    camera_names = ['top']
    state_dim = 8

    # fixed parameters
    backbone = 'resnet18'
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    policy_config = {'lr': args['lr'],
                     'num_queries': chunk_size,
                     'kl_weight': args['kl_weight'],
                     'hidden_dim': args['hidden_dim'],
                     'dim_feedforward': args['dim_feedforward'],
                     'lr_backbone': args['lr'],
                     'backbone': backbone,
                     'enc_layers': enc_layers,
                     'dec_layers': dec_layers,
                     'nheads': nheads,
                     'camera_names': camera_names,
                     'state_dim': state_dim
                     }

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'camera_names': camera_names,
        'save_epoch': save_epoch,
        'resume_ckpt': resume_ckpt
    }

    task_config = ['RH20T_cfg1','RH20T_cfg2','RH20T_cfg3','RH20T_cfg4','RH20T_cfg5','RH20T_cfg6','RH20T_cfg7']
    train_dataset = RH20TPretrain(dataset_root, task_config, 'train', num_input=1, horizon=1+chunk_size, top_down_view=True, selected_tasks=[task_name])
    val_dataset = RH20TPretrain(dataset_root, task_config, 'val', num_input=1, horizon=1+chunk_size, top_down_view=True, selected_tasks=[task_name])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=20, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=20, collate_fn=collate_fn)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

def forward_pass(data, policy, device):
    image_data = data['input_frame_list']
    qpos_data = data['input_frame_tcp_normalized']
    action_data = data['target_frame_tcp_normalized']
    is_pad = data['padding_mask']
    image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_config = config['policy_config']

    set_seed(seed)    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy = ACTPolicy(policy_config)

    if config['resume_ckpt'] is not None:
        policy.load_state_dict(torch.load(config['resume_ckpt'], map_location = device))
        print('Loaded checkpoint from %s' % (config['resume_ckpt']))
    
    optimizer = policy.configure_optimizers()

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            with tqdm(val_dataloader) as pbar:
                for data in pbar:
                    forward_dict = forward_pass(data, policy, device)
                    epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        num_steps = len(train_dataloader)
        with tqdm(train_dataloader) as pbar:
            for data in pbar:
                forward_dict = forward_pass(data, policy, device)
                # backward
                loss = forward_dict['loss']
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[num_steps*epoch:num_steps*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)
        if epoch % config["save_epoch"] == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--dataset_root', action='store', type=str, help='dataset_root', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--save_epoch', action='store', type=int, help='save frequency (epoch)', default=10, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--resume_ckpt', action='store', type=str, help='checkpoint to resume training', default=None, required=False)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    
    main(vars(parser.parse_args()))
