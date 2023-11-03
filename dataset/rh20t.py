''' Demo for loading and processing RH20T data.
    Author: chenxi-wang
'''

import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset

TO_TENSOR_KEYS = ['input_frame_list', 'input_frame_tcp_normalized', 'target_frame_tcp_normalized', 'padding_mask']
IN_HAND_CAM_IDS = {'RH20T_cfg1': ['cam_043322070878'],
                   'RH20T_cfg2': ['cam_104422070042'],
                   'RH20T_cfg3': ['cam_045322071843'],
                   'RH20T_cfg4': ['cam_045322071843'],
                   'RH20T_cfg5': ['cam_104422070042', 'cam_135122079702'],
                   'RH20T_cfg6': ['cam_135122070361', 'cam_135122075425'],
                   'RH20T_cfg7': ['cam_135122070361', 'cam_135122075425']}
TOP_DOWN_CAM_IDS = {'RH20T_cfg1': ['cam_750612070851', 'cam_039422060546', 'cam_750612070853'],
                    'RH20T_cfg2': ['cam_f0461559', 'cam_037522062165', 'cam_104122061850'],
                    'RH20T_cfg3': ['cam_038522062288', 'cam_104122062295'],
                    'RH20T_cfg4': ['cam_038522062288', 'cam_104122062295'],
                    'RH20T_cfg5': ['cam_037522062165'],
                    'RH20T_cfg6': ['cam_104122061330'],
                    'RH20T_cfg7': ['cam_104122061330']}

class RH20TDataset(Dataset):
    def __init__(self, root, task_config_list, split='train', num_input=1, horizon=1+20, image_size=(360,640), image_mean=[0.485,0.456,0.406], image_std=[0.229,0.224,0.225], dict_path='dataset/rh20t_cleaned_data.json', frame_sample_step=1, top_down_view=False, selected_tasks=None):
        assert split in ['train', 'val', 'all']
        self.root = root
        self.split = split
        self.num_input = num_input
        self.horizon = horizon
        self.image_size = image_size
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.top_down_view = top_down_view
        
        self.input_task_ids = []
        self.input_cam_ids = []
        self.input_task_configs = []
        self.target_frame_ids = []
        self.padding_mask_list = []
        
        with open(dict_path, 'r') as f:
            data_dict = json.load(f)
        self.task_ids, self.cam_ids, self.task_configs = load_all_tasks(root, task_config_list, data_dict, split, top_down_view, selected_tasks)
        num_tasks = len(self.task_ids)
        print('#tasks:', num_tasks)

        for i in tqdm(range(num_tasks), desc='loading data samples...'):
            task_id, cam_id, task_config = self.task_ids[i], self.cam_ids[i], self.task_configs[i]
            meta_path = os.path.join(self.root, task_config, task_id, 'metadata.json')
            metadata = json.load(open(meta_path))
            frame_ids = data_dict[task_config][task_id][cam_id]
            frame_ids = [x for x in frame_ids if x <= metadata['finish_time']]
            target_frame_ids, padding_mask_list = self._get_input_output_frame_id_lists(frame_ids, num_input=num_input, horizon=horizon, frame_sample_step=frame_sample_step)
            self.target_frame_ids += target_frame_ids
            self.padding_mask_list += padding_mask_list
            self.input_task_ids += [task_id] * len(target_frame_ids)
            self.input_cam_ids += [cam_id] * len(target_frame_ids)
            self.input_task_configs += [task_config] * len(target_frame_ids)


    def __len__(self):
        return len(self.target_frame_ids)
    
    def _get_input_output_frame_id_lists(self, frame_id_list, num_input=1, horizon=1+20, frame_sample_step=1):
        target_frame_ids = []
        padding_mask_list = []

        if len(frame_id_list) < horizon:
            # padding
            frame_id_list = frame_id_list + frame_id_list[-1:] * (horizon-len(frame_id_list))

        # padding for the first (num_input-1) frames
        frame_id_list = frame_id_list[0:1] * (num_input-1) * frame_sample_step + frame_id_list
        for i in range(len(frame_id_list)-int(num_input*frame_sample_step)):
            cur_target_frame_ids = frame_id_list[i:i+horizon*frame_sample_step:frame_sample_step]
            padding_mask = np.zeros(horizon, dtype=bool)
            if len(cur_target_frame_ids) < horizon:
                cur_target_frame_ids += [frame_id_list[-1]] * (horizon - len(cur_target_frame_ids))
                padding_mask[len(cur_target_frame_ids):] = 1
            target_frame_ids.append(cur_target_frame_ids)
            padding_mask_list.append(padding_mask)

        return target_frame_ids, padding_mask_list

    def _clip_tcp(self, tcp_list):
        ''' tcp_list: [T, 8]'''
        tcp_list[:,0] = np.clip(tcp_list[:,0], -0.64, 0.64)
        tcp_list[:,1] = np.clip(tcp_list[:,1], -0.64, 0.64)
        tcp_list[:,2] = np.clip(tcp_list[:,2], 0, 1.28)
        tcp_list[:,7] = np.clip(tcp_list[:,7], 0, 0.11)
        return tcp_list

    def _normalize_tcp(self, tcp_list):
        ''' tcp_list: [T, 8]'''
        if self.top_down_view:
            trans_min, trans_max = np.array([-0.35, -0.35, 0]), np.array([0.35, 0.35, 0.7])
        else:
            trans_min, trans_max = np.array([-0.64, -0.64, 0]), np.array([0.64, 0.64, 1.28])
        max_gripper_width = 0.11 # meter
        tcp_list[:,:3] = (tcp_list[:,:3] - trans_min) / (trans_max - trans_min) * 2 - 1
        tcp_list[:,7] = tcp_list[:,7] / max_gripper_width * 2 - 1
        return tcp_list

    def __getitem__(self, index):
        task_id = self.input_task_ids[index]
        target_frame_ids = self.target_frame_ids[index]
        padding_mask = self.padding_mask_list[index]
        cam_id = self.input_cam_ids[index]
        task_config = self.input_task_configs[index]

        # load input rgbs
        input_frame_list = []
        point_mask_list = []
        for input_frame_id in target_frame_ids[:self.num_input]:
            color_path = os.path.join(self.root, task_config, task_id, cam_id, 'color', '%d.jpg'%input_frame_id)
            color = np.array(Image.open(color_path).resize(self.image_size), dtype=np.float32) / 255.0
            # imagenet normalization
            color = (color - self.image_mean) / self.image_std
            input_frame_list.append(color)

        gripper_path = os.path.join(self.root, task_config, task_id, 'transformed', 'gripper.npy')
        tcp_path = os.path.join(self.root, task_config, task_id, 'transformed', 'tcp.npy')

        # load input and target gripper pose
        tcp_list = np.load(tcp_path, allow_pickle=True)[()][cam_id[4:]]
        target_frame_tcp_list = []
        i, p = 0, 0
        while i < len(tcp_list):
            while p < self.horizon and tcp_list[i]['timestamp'] == target_frame_ids[p]:
                target_frame_tcp_list.append(tcp_list[i]['tcp'].astype(np.float32))
                p += 1
            if p == self.horizon:
                break
            i += 1
        assert p == self.horizon, 'p:%d, input:%d' % (p, self.horizon)

        target_frame_tcp_list = np.array(target_frame_tcp_list, dtype=np.float32)


        # get gripper label
        gripper_list = np.load(gripper_path, allow_pickle=True)[()][cam_id[4:]]
        target_gripper_width_list = []
        for i,fid in enumerate(target_frame_ids):
            if i < self.num_input:
                gripper_command = gripper_list[fid]['gripper_info']
            else:
                gripper_command = gripper_list[fid]['gripper_command']
            gripper_width = gripper_command[0] / 1000. # transform mm into m
            target_gripper_width_list.append(gripper_width)
        target_gripper_width_list = np.array(target_gripper_width_list, dtype=np.float32)[:,np.newaxis]
        target_frame_tcp_list = np.concatenate([target_frame_tcp_list, target_gripper_width_list], axis=-1)

        # get normalized tcp
        target_frame_tcp_list = np.array(target_frame_tcp_list, dtype=np.float32)
        target_frame_tcp_list = self._clip_tcp(target_frame_tcp_list)
        target_frame_tcp_normalized = self._normalize_tcp(target_frame_tcp_list.copy())

        # split data
        input_frame_tcp_list = target_frame_tcp_list[:self.num_input]
        target_frame_tcp_list = target_frame_tcp_list[self.num_input:]
        input_frame_tcp_normalized = target_frame_tcp_normalized[:self.num_input]
        target_frame_tcp_normalized = target_frame_tcp_normalized[self.num_input:]
        padding_mask = padding_mask[self.num_input:]

        # make input
        input_frame_list = np.stack(input_frame_list, axis=0)[:,np.newaxis] # (..., 360, 640, 3)
        input_frame_list = np.transpose(input_frame_list, axes=[0,1,4,2,3]) # (..., 3, 360, 480)

        if self.num_input == 1:
            input_frame_list = input_frame_list[0]
            input_frame_tcp_list = input_frame_tcp_list[0]
            input_frame_tcp_normalized = input_frame_tcp_normalized[0]

        ret_dict = {'input_frame_list': input_frame_list,
                    'input_frame_tcp_list': input_frame_tcp_list,
                    'input_frame_tcp_normalized': input_frame_tcp_normalized,
                    'target_frame_tcp_list': target_frame_tcp_list,
                    'target_frame_tcp_normalized': target_frame_tcp_normalized,
                    'padding_mask': padding_mask,
                    'task_id': task_id,
                    'target_frame_ids': target_frame_ids,
                    'cam_id': cam_id}

        return ret_dict
        

def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in TO_TENSOR_KEYS:
                ret_dict[key] = collate_fn([d[key] for d in batch])
            else:
                ret_dict[key] = [d[key] for d in batch]
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))

def load_all_tasks(task_root, task_configs, data_dict, split='train', top_down_view=False, selected_tasks=None):
    assert split in ['train', 'val', 'all']
    task_ids = []
    cam_ids = []
    config_ids = []

    def _get_scene_meta(scene_dir):
        meta_path = os.path.join(scene_dir, 'metadata.json')
        metadata = json.load(open(meta_path))
        return metadata

    for task_config in task_configs:
        cur_task_ids = sorted(data_dict[task_config].keys())

        if split == 'train':
            cur_task_ids = [tid for tid in cur_task_ids if 'scene_0010' not in tid]
        elif split == 'val':
            cur_task_ids = [tid for tid in cur_task_ids if 'scene_0010' in tid]

        for task_id in cur_task_ids:
            if selected_tasks is not None and task_id[:9] not in selected_tasks:
                continue
            scene_dir = os.path.join(task_root, task_config, task_id)
            metadata = _get_scene_meta(scene_dir)
            if 'rating' not in metadata or metadata['rating'] <= 1:
                continue
            cur_cam_ids = sorted(data_dict[task_config][task_id])
            for cam_id in cur_cam_ids:
                if top_down_view and cam_id not in TOP_DOWN_CAM_IDS[task_config]:
                    continue
                task_ids.append(task_id)
                cam_ids.append(cam_id)
                config_ids.append(task_config)

    return task_ids, cam_ids, config_ids

def parse_action_preds(action_preds, max_gripper_width=0.11, top_down_view=False):
    ''' logits: numpy.ndarray, [B,T,8]
    '''
    if top_down_view:
        trans_min, trans_max = np.array([-0.35, -0.35, 0]), np.array([0.35, 0.35, 0.7])
    else:
        trans_min, trans_max = np.array([-0.64, -0.64, 0]), np.array([0.64, 0.64, 1.28])

    trans_preds = action_preds[...,0:3]
    trans_preds = (trans_preds + 1) / 2.0
    trans_preds = trans_preds * (trans_max - trans_min) + trans_min

    quat_preds = action_preds[...,3:7]
    quat_preds /=  np.linalg.norm(quat_preds, axis=2, keepdims=True) + 1e-6

    gripper_width_preds = action_preds[...,7:8]
    gripper_width_preds = (gripper_width_preds + 1) / 2.0
    gripper_width_preds = gripper_width_preds * max_gripper_width

    action_preds = np.concatenate([trans_preds, quat_preds, gripper_width_preds], axis=-1)

    return action_preds
