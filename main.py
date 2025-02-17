import os
import torch
import argparse
from torch.utils.data import DataLoader
from data import ImitationDataset, temporal_split, collate_fn
from models import WaypointPredictor
from train import Trainer
from utils import load_config, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-dir', required=True)
    args = parser.parse_args()
    
    # Load config
    config = load_config(os.path.join(args.config, "default.yaml"))
    set_seed(config['seed'])
    
    # Initialize dataset
    dataset = ImitationDataset(
        objects_csv=os.path.join(args.data_dir, "cametra_interface_output.csv"),
        lanes_csv=os.path.join(args.data_dir, "cametra_interface_lanes_output.csv"),
        imu_json=os.path.join(args.data_dir, "imu_data.json"),
        waypoints_npy=os.path.join(args.data_dir, "waypoints.npy"),
        mode='train',
        noise_std=config['noise_std'],
        max_ts_diff=config['max_ts_diff']
    )
    
    # Split dataset
    train_set, val_set = temporal_split(dataset, config['val_split'])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config['batch_size'],
        collate_fn=collate_fn
    )
    
    # Initialize model and trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaypointPredictor(config)
    trainer = Trainer(model, train_loader, val_loader, config, device)
    # Training loop
    trainer.run(config['epochs'])

if __name__ == '__main__':
    main()