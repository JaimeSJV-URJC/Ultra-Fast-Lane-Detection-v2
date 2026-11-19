import torch, os
from utils.common import merge_config, get_model
from evaluation.eval_wrapper import eval_lane
import torch
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    cfg.distributed = distributed
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    net = get_model(cfg)

    state_dict = torch.load(cfg.test_model, map_location = 'cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict = True)

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])

    if not cfg.test_work_dir:
        cfg.test_work_dir = os.path.join(os.path.dirname(os.path.abspath(cfg.test_model)), 'test')
    if not os.path.exists(cfg.test_work_dir):
        os.makedirs(cfg.test_work_dir)

    eval_lane(net, cfg)