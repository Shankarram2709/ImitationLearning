import csv
import json
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
from utils import find_nearest_timestamp

class ImitationDataset(Dataset):
    def __init__(self, objects_csv, lanes_csv, imu_json, waypoints_npy, 
                 norm_stats=None, mode='train', noise_std=0.05, max_ts_diff=1e5):
        self.mode = mode
        self.noise_std = noise_std
        
        # Load IMU data (reference timestamps)
        with open(imu_json) as f:
            self.imu_data = json.load(f)
        self.imu_ts = sorted([int(ts) for ts in self.imu_data.keys()])
        
        # Load objects and lanes
        self.objects_ts, self.objects = self._load_objects(objects_csv)
        self.lanes_ts, self.lanes = self._load_lanes(lanes_csv)
        self.waypoints = np.load(waypoints_npy)
        self.max_ts_diff = max_ts_diff
        self.norm_stats = norm_stats or self._compute_stats()

    def _load_objects(self, path):
        ts_list = []
        data = {}
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = int(float(row['name']))
                    lane_assoc = int(row['lane_association'])
                    lane_assoc_onehot = torch.zeros(4, dtype=torch.float32)
                    lane_assoc_onehot[lane_assoc] = 1.0
                    feat = torch.tensor([
                        float(row['lat_dist']),
                        float(row['long_dist']),
                        float(row['abs_vel_x']),
                        float(row['abs_vel_z']),
                        float(row['is_cipv']),
                        float(row['age']) / float(max(row['age']))
                    ], dtype=torch.float32)
                    feat = torch.cat([feat, lane_assoc_onehot, torch.tensor([1.0])])
                    data.setdefault(ts, []).append(feat)
                    ts_list.append(ts)
                except KeyError as e:
                    #continue
                    print(f"Skipping row {row_idx}: Missing column {e}")
        return sorted(ts_list), data

    def _load_lanes(self, path):
        ts_list = []
        data = {}
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = int(float(row['frame_id']))
                    feat = torch.tensor([
                        float(row['fDistanceMeter']),
                        float(row['fYawAngleRad']),
                        float(row['polynomial[0]']),
                        float(row['polynomial[1]']),
                        float(row['polynomial[2]'])
                    ], dtype=torch.float32)
                    feat = torch.cat([feat, torch.tensor([1.0])])
                    data.setdefault(ts, []).append(feat)
                    ts_list.append(ts)
                except KeyError as e:
                    print(f"Skipping row {row_idx}: Missing column {e}")
        return sorted(ts_list), data

    def _compute_stats(self):
        stats = {'imu': {'mean': None, 'std': None}, 
                'object': {'mean': None, 'std': None},
                'lane': {'mean': None, 'std': None}}
        
        # IMU stats- (4 features)
        imu_values = []
        for ts in self.imu_ts:
            imu = self.imu_data[str(ts)]
            imu_values.append([float(imu['vf']), float(imu['ax']),
                               float(imu['yaw']), float(imu['wz'])])
        stats['imu']['mean'] = torch.tensor(np.nanmean(imu_values, axis=0), dtype=torch.float32)
        stats['imu']['std'] = torch.tensor(np.nanstd(imu_values, axis=0), dtype=torch.float32)
        
        # Object stats- (10 features)
        obj_values = []
        for ts, feats in self.objects.items():
            for feat in feats:
                if feat[-1] == 1.0:  # Check validity flag
                    obj_values.append(feat[:-1].numpy())
        stats['object']['mean'] = torch.tensor(np.nanmean(obj_values, axis=0), dtype=torch.float32) if obj_values else torch.zeros(11, dtype=torch.float32)
        stats['object']['std'] = torch.tensor(np.nanstd(obj_values, axis=0), dtype=torch.float32) if obj_values else torch.ones(11, dtype=torch.float32)
        
        # Lane stats- (5 features)
        lane_values = []
        for ts, feats in self.lanes.items():
            for feat in feats:
                if feat[-1] == 1.0:
                    lane_values.append(feat[:-1].numpy())
        stats['lane']['mean'] = torch.tensor(np.nanmean(lane_values, axis=0), dtype=torch.float32) if lane_values else torch.zeros(6, dtype=torch.float32)
        stats['lane']['std'] = torch.tensor(np.nanstd(lane_values, axis=0), dtype=torch.float32) if lane_values else torch.ones(6, dtype=torch.float32)

        # Handle NaNs
        for modality in stats:
            stats[modality]['mean'] = torch.nan_to_num(stats[modality]['mean'], nan=0.0)
            stats[modality]['std'] = torch.nan_to_num(stats[modality]['std'], nan=1.0)
            stats[modality]['std'] = torch.clamp(stats[modality]['std'], min=1e-6)
        
        return stats

    def _normalize(self, tensor, modality):
        return (tensor - self.norm_stats[modality]['mean']) / self.norm_stats[modality]['std']

    def __getitem__(self, idx):
        imu_ts = self.imu_ts[idx]
        
        # IMU with noise
        imu = torch.tensor([
            float(self.imu_data[str(imu_ts)]['vf']),
            float(self.imu_data[str(imu_ts)]['ax']),
            float(self.imu_data[str(imu_ts)]['yaw']),
            float(self.imu_data[str(imu_ts)]['wz'])
        ], dtype=torch.float32)
        #use if noise addition might be required
        # if self.mode == 'train':
        #     imu += torch.randn_like(imu) * self.noise_std
        
        imu = self._normalize(imu, 'imu')
        objects = [self._normalize(feat, 'object') for feat in self.objects]
        lanes = [self._normalize(feat, 'lane') for feat in self.lanes]
        # Find nearest object timestamp
        obj_ts = find_nearest_timestamp(imu_ts, self.objects_ts, self.max_ts_diff)
        objects = self.objects.get(obj_ts, [torch.zeros(11)]) if obj_ts else [torch.zeros(11)]
        
        # Find nearest lane timestamp
        lane_ts = find_nearest_timestamp(imu_ts, self.lanes_ts, self.max_ts_diff)
        lanes = self.lanes.get(lane_ts, [torch.zeros(6)]) if lane_ts else [torch.zeros(6)]
        
        return {
            'imu': imu,
            'objects': torch.stack(objects),
            'lanes': torch.stack(lanes),
            'waypoints': torch.tensor(self.waypoints[idx], dtype=torch.float32)
        }

    def __len__(self):
        return len(self.imu_ts)

def collate_fn(batch):
    imu = torch.stack([item['imu'].to(torch.float32) for item in batch])
    # Process objects
    obj_list = [item['objects'] for item in batch]
    obj_padded = pad_sequence(obj_list, batch_first=True).to(torch.float32)
    obj_mask = torch.zeros(obj_padded.shape[:2], dtype=torch.bool)
    for i, tensor in enumerate(obj_list):
        obj_mask[i, :tensor.size(0)] = True
    
    # Process lanes
    lane_list = [item['lanes'] for item in batch]
    lane_padded = pad_sequence(lane_list, batch_first=True).to(torch.float32)
    lane_mask = torch.zeros(lane_padded.shape[:2], dtype=torch.bool)
    for i, tensor in enumerate(lane_list):
        lane_mask[i, :tensor.size(0)] = True
    
    return {
        'imu': imu,
        'objects': obj_padded,
        'objects_mask': obj_mask,
        'lanes': lane_padded,
        'lanes_mask': lane_mask,
        'waypoints': torch.stack([item['waypoints'] for item in batch])
    }

def temporal_split(dataset, val_ratio=0.2):
    total = len(dataset)
    split_idx = int(total * (1 - val_ratio))
    return Subset(dataset, list(range(split_idx))), Subset(dataset, list(range(split_idx, total)))