import yaml
import torch
import bisect

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_nearest_timestamp(target, candidates, max_diff):
    #timestamp sampling between imu and detected info from camera
    idx = bisect.bisect_left(candidates, target)
    if idx == 0:
        nearest = candidates[0]
    elif idx == len(candidates):
        nearest = candidates[-1]
    else:
        before = candidates[idx-1]
        after = candidates[idx]
        nearest = before if (target - before) <= (after - target) else after
    return nearest if abs(nearest - target) <= float(max_diff) else None