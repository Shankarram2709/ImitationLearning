import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, input_dim, attn_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, attn_dim)
        self.key = nn.Linear(input_dim, attn_dim)
        self.value = nn.Linear(input_dim, attn_dim)
        self.scale = nn.Parameter(torch.tensor(attn_dim ** -0.5))
        
    def forward(self, x, mask):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.bmm(Q, K.transpose(1,2)) * self.scale
        scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, V).mean(dim=1)

class WaypointPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.obj_attn = TemporalAttention(10, config['attn_dim'])
        self.lane_attn = TemporalAttention(5, config['attn_dim'])
        self.context_proj = nn.Linear(config['fusion_hidden'], 2*config['gru_hidden'])

        self.fusion = nn.Sequential(
            nn.Linear(2 * config['attn_dim'] + 4, config['fusion_hidden']),
            nn.ReLU(),
            nn.BatchNorm1d(config['fusion_hidden']), 
            nn.Dropout(config['dropout']),
            nn.LayerNorm(256),
            nn.Linear(256, config['fusion_hidden']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.LayerNorm(config['fusion_hidden'])
        )
        
        self.gru = nn.GRU(
            input_size=2,
            hidden_size=config['gru_hidden'],
            num_layers=2,
            dropout=config['dropout'],
            batch_first=True
        )
        self.waypoint_head = nn.Sequential(
            nn.Linear(config['gru_hidden'], 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, imu, objects, lanes, obj_mask, lane_mask):
        obj_pool = self.obj_attn(objects[:, :, :-1], obj_mask)
        lane_pool = self.lane_attn(lanes[:, :, :-1], lane_mask)
        #from IPython import embed;embed()
        fused = torch.cat([obj_pool, lane_pool, imu], dim=1)
        context = self.fusion(fused)
        
        h0 = self.context_proj(context).view(2, -1, 128) #config['gru_hidden']
        waypoints = []
        x = torch.zeros(imu.size(0), 1, 2, device=imu.device)
        for _ in range(4):
            out, h0 = self.gru(x, h0)
            wp = self.waypoint_head(out[:, -1])
            waypoints.append(wp)
            x = wp.unsqueeze(1)
            
        return torch.stack(waypoints, dim=1)