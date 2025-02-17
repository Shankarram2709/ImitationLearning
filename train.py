import torch
from torch.utils.tensorboard import SummaryWriter
from utils import load_config

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optim = torch.optim.AdamW(
            model.parameters(), 
            lr=float(config['lr']), 
            weight_decay=float(config['weight_decay'])
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=0.5, patience=3)
        self.weights = torch.tensor(config['waypoint_weights'], device=self.device)
        self.writer = SummaryWriter()
    
    # def kinematic_loss(self, pred_waypoints):
    #     """Penalize loss related to acceleration if necessary"""
    #     # pred_waypoints shape: (batch_size, num_waypoints, 2)
    #     accel = pred_waypoints[:, 1:, :] - pred_waypoints[:, :-1, :]  # Δvelocity between steps
    #     return torch.mean(accel ** 2)  # MSE of acceleration

    def compute_loss(self, pred, target):
        # mse_loss = (self.weights * (pred - target).pow(2).mean(dim=-1)).mean()
        # k_loss = self.kinematic_loss(pred) * 0.05  # Weighted to balance MSE and smoothness
        # return mse_loss + k_loss
        return (self.weights * (pred - target).pow(2).mean(dim=-1)).mean()

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            self.optim.zero_grad()
            # Move data to device
            imu = batch['imu'].to(self.device).to(torch.float32)
            objects = batch['objects'].to(self.device).to(torch.float32)
            lanes = batch['lanes'].to(self.device).to(torch.float32)
            obj_mask = batch['objects_mask'].to(self.device)
            lane_mask = batch['lanes_mask'].to(self.device)
            targets = batch['waypoints'].to(self.device)
            
            # Forward pass
            preds = self.model(imu, objects, lanes, obj_mask, lane_mask)
            loss = self.compute_loss(preds, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optim.step()
            
            total_loss += loss.item() * imu.size(0)
        return total_loss / len(self.train_loader.dataset)

    def validate(self):
        self.model.eval()
        total_loss = ade_sum = fde_sum = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                imu = batch['imu'].to(self.device)
                objects = batch['objects'].to(self.device)
                lanes = batch['lanes'].to(self.device)
                obj_mask = batch['objects_mask'].to(self.device)
                lane_mask = batch['lanes_mask'].to(self.device)
                targets = batch['waypoints'].to(self.device)
                
                preds = self.model(imu, objects, lanes, obj_mask, lane_mask)
                loss = self.compute_loss(preds, targets)
                total_loss += loss.item() * imu.size(0)
                
                # Compute metrics
                ade = torch.norm(preds - targets, dim=-1).mean()
                fde = torch.norm(preds[:, -1] - targets[:, -1], dim=-1).mean()
                ade_sum += ade.item() * imu.size(0)
                fde_sum += fde.item() * imu.size(0)
                
        return {
            'loss': total_loss / len(self.val_loader.dataset),
            'ade': ade_sum / len(self.val_loader.dataset),
            'fde': fde_sum / len(self.val_loader.dataset)
        }

    def run(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step(val_metrics['loss'])
            
            # Logging
            self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_metrics['loss']}, epoch)
            self.writer.add_scalar('ADE', val_metrics['ade'], epoch)
            self.writer.add_scalar('FDE', val_metrics['fde'], epoch)
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"ADE: {val_metrics['ade']:.4f} | "
                  f"FDE: {val_metrics['fde']:.4f}")