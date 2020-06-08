import argparse

from Train.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='train')
parser.add_argument('--cfg', type=str, default='Configs/GIFT-stage1.yaml')
parser.add_argument('--det_cfg', type=str, default='Configs/eval/superpoint_det.yaml')
parser.add_argument('--desc_cfg', type=str, default='Configs/eval/gift_pretrain_desc.yaml')
parser.add_argument('--match_cfg', type=str, default='Configs/eval/match_v2.yaml')
flags = parser.parse_args()


def train():
    trainer = Trainer(flags.cfg)
    trainer.train()


if __name__ == "__main__":
    name2func = {
        'train': train,
    }
    name2func[flags.task]()
