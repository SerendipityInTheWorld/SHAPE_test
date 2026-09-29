# =============================================================================
# test_full.py —— 独立全指标评估入口（不改动 val.py / test.py）
# 在目标域测试集上计算逐类 Dice / ASD / HD95 / Jaccard / Sensitivity / Specificity，
# 额外统计“幻影类”（GT 无该类但预测出现）与“纯背景样本”，结果写入 JSON。
# 用法（示例，MMWHS CT 有标签 -> MR 测试，权重在 checkpoints/CT/foo/shape_best_model_*.pth）：
#   python test_full.py --mode CT --gpu 0 --dino_size s \
#       --data_path E:/.../数据集/MMWHS_2D \
#       --pth checkpoints/CT/foo/shape_best_model_0.500000.pth
# 指标口径：
#   背景类（label 0）不参与；仅统计类 1..num_classes-1。
#   - 幻影类：GT 该样本无类 c 但预测出现 -> Dice/Jaccard=0 有效计入该类均值；
#     ASD/HD95/Sensitivity 记 NaN（无 GT 表面，不参与该类均值）；Specificity 正常计入。
#   - 纯背景样本：GT 全为 0；预测也全 0 -> n_pure_bg_perfect；预测含类 -> 计 n_pure_bg_with_pred
#     且各预测类按幻影规则计 0 分。
#   - 空洞样本（GT 与预测均无类 c）：跳过该类该样本。
#   Dice 为百分数（dc*100，与官方一致）；Jaccard/Sensitivity/Specificity 为 0-1；
#   ASD/HD95 单位为像素。
# =============================================================================
import os
import json
import time
import random
import argparse
import configparser
import logging
from datetime import timedelta
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import Getfile
from network.SHAPE_net import SHAPE
from utils import binary


def get_args():
    parser = argparse.ArgumentParser(description='Full-metric evaluation (independent of val.py/test.py)')
    parser.add_argument('--data_path', type=str, default='../data2D', help='Dataset root')
    parser.add_argument('--pth', type=str, required=True, help='Decoder weights file (.pth)')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Override target test dir (default: config.ini test_dir[1])')
    parser.add_argument('--json_out', type=str, default=None,
                        help='Output JSON path (default: <pth dir>/eval_<target>_<timestamp>.json)')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--model_type', type=str, default='shape')
    parser.add_argument('--mode', type=str, default='CT', help='Source mode: CT/MR/ABCT/ABMR')
    parser.add_argument('--classes', '-c', type=int, default=5, help='Number of classes (incl. background)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers (Windows: 0)')
    parser.add_argument('--use_hfm', action='store_true')
    parser.add_argument('--use_selector', action='store_true')
    parser.add_argument('--use_refinement', action='store_true')
    parser.add_argument('--use_pseudo_labels', action='store_true')
    parser.add_argument('--hpe_fusion_alpha', type=float, default=0.25)
    parser.add_argument('--selector_initial_k', type=float, default=0.1)
    parser.add_argument('--sap_threshold_percentile', type=float, default=50.0)
    parser.add_argument('--hpe_use_historical_stats', action='store_true')
    parser.add_argument('--pure_tao', type=float, default=1)
    parser.add_argument("--repo_dir", type=str, default="dinov3")
    parser.add_argument("--dino_ckpt", type=str, default="dinov3_checkpoint")
    parser.add_argument("--dino_size", type=str, default="s", choices=["s+", "s"])
    return parser.parse_args()


def infer_modes_from_path(source_modality):
    if source_modality == 'CT':
        target_modality = 'MR'
    elif source_modality == 'MR':
        target_modality = 'CT'
    elif source_modality == 'ABCT':
        target_modality = 'ABMR'
    elif source_modality == 'ABMR':
        target_modality = 'ABCT'
    else:
        raise ValueError('Unknown source modality {!r}'.format(source_modality))
    return source_modality, target_modality


def load_config(mode, config_path='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_path)
    test_dirs = config.get(mode, 'test_dir').split(", ")
    label_intensities = tuple(map(float, config.get(mode, 'label_intensities').split(', ')))
    return test_dirs, label_intensities


METRICS = ['Dice', 'ASD', 'HD95', 'Jaccard', 'Sensitivity', 'Specificity']


def evaluate_full(dataloader, model, num_classes):
    """逐样本逐类收集指标；返回聚合 dict（含幻影类/纯背景统计）。"""
    model.eval()
    stats = {}
    for c in range(1, num_classes):
        stats[c] = {m: [] for m in METRICS}
        stats[c]['n_gt'] = 0
        stats[c]['n_pred'] = 0
        stats[c]['phantom_n'] = 0

    n_samples = 0
    n_pure_bg = 0
    n_pure_bg_perfect = 0
    n_pure_bg_with_pred = 0

    with torch.no_grad():
        for batch in dataloader:
            xt = batch['s'].cuda()
            xt_labels = batch['label'].cuda()
            logits = model.inference(xt)
            out = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            out_np = out.cpu().numpy()
            gt_np = xt_labels.cpu().numpy()

            for s in range(out_np.shape[0]):
                pred = out_np[s]
                gt = gt_np[s].squeeze(0)
                n_samples += 1

                is_pure_bg = (gt.sum() == 0)
                if is_pure_bg:
                    n_pure_bg += 1

                for c in range(1, num_classes):
                    p = (pred == c)
                    g = (gt == c)
                    has_p = p.sum() > 0
                    has_g = g.sum() > 0
                    if not has_p and not has_g:
                        continue

                    stats[c]['Dice'].append(_safe(binary.dc, p, g) * 100.0)
                    stats[c]['Jaccard'].append(_safe(binary.jc, p, g))

                    if has_g:
                        stats[c]['n_gt'] += 1
                    else:
                        stats[c]['phantom_n'] += 1
                    if has_p:
                        stats[c]['n_pred'] += 1

                    if has_g:
                        stats[c]['Sensitivity'].append(_safe(binary.sensitivity, p, g))
                    else:
                        stats[c]['Sensitivity'].append(np.nan)

                    if has_g and has_p:
                        stats[c]['ASD'].append(_safe(binary.asd, p, g))
                        stats[c]['HD95'].append(_safe(binary.hd95, p, g))
                    else:
                        stats[c]['ASD'].append(np.nan)
                        stats[c]['HD95'].append(np.nan)

                    stats[c]['Specificity'].append(_safe(binary.specificity, p, g))

                if is_pure_bg:
                    if pred.sum() == 0:
                        n_pure_bg_perfect += 1
                    else:
                        n_pure_bg_with_pred += 1

    per_class = []
    mean = {}
    for c in range(1, num_classes):
        entry = {'class': int(c), 'n_valid_samples': int(len(stats[c]['Dice']))}
        for m in METRICS:
            arr = np.asarray(stats[c][m], dtype=np.float64)
            if arr.size == 0:
                entry[m] = None
                entry[m + '_n'] = 0
            else:
                valid = arr[np.isfinite(arr)]
                entry[m] = float(np.mean(valid)) if valid.size else None
                entry[m + '_n'] = int(valid.size)
        entry['n_gt'] = int(stats[c]['n_gt'])
        entry['n_pred'] = int(stats[c]['n_pred'])
        entry['phantom_n'] = int(stats[c]['phantom_n'])
        per_class.append(entry)

    for m in METRICS:
        vals = [e[m] for e in per_class if e[m] is not None]
        mean[m] = float(np.mean(vals)) if vals else None

    return {
        'n_samples': n_samples,
        'n_pure_bg': n_pure_bg,
        'n_pure_bg_perfect': n_pure_bg_perfect,
        'n_pure_bg_with_pred': n_pure_bg_with_pred,
        'per_class': per_class,
        'mean': mean,
    }


def _safe(fn, p, g):
    try:
        return fn(p, g)
    except RuntimeError:
        return np.nan


def main():
    args = get_args()
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    random.seed(8888)
    torch.manual_seed(8888)
    np.random.seed(8888)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger('main_logger')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device: {} (visible physical GPU {})'.format(device, args.gpu))

    source_mode, target_mode = infer_modes_from_path(args.mode)
    test_dirs, label_intensities = load_config(source_mode)
    target_test_dir = args.test_dir if args.test_dir is not None else test_dirs[1]
    logger.info('Source={} -> Target={}; target test dir: {}'.format(source_mode, target_mode, target_test_dir))

    if args.dino_size == "s+":
        dino_ckpt = os.path.join(args.dino_ckpt, 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth')
        backbone = torch.hub.load(repo_or_dir=args.repo_dir, model='dinov3_vits16plus', source='local',
                                  weights=dino_ckpt, pretrained=True)
    else:
        dino_ckpt = os.path.join(args.dino_ckpt, 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
        backbone = torch.hub.load(repo_or_dir=args.repo_dir, model='dinov3_vits16', source='local',
                                  weights=dino_ckpt, pretrained=True)
    logger.info('Loaded DINOv3 ViT-{}/16 backbone.'.format(args.dino_size.upper()))

    model = SHAPE(backbone=backbone, nclass=args.classes, args=args).to(device)

    if not os.path.exists(args.pth):
        logger.error('Checkpoint not found: {}'.format(args.pth))
        return

    checkpoint = torch.load(args.pth, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        checkpoint = checkpoint['model']
    elif not isinstance(checkpoint, dict):
        logger.error('Checkpoint is not a state_dict.')
        return

    weights_to_load = {}
    has_decoder_key = any('.decoder.' in k or k.startswith('decoder.') for k in checkpoint.keys())
    has_module_prefix = any(k.startswith('module.') for k in checkpoint.keys())
    if not has_decoder_key and not has_module_prefix:
        weights_to_load = dict(checkpoint)
    else:
        for k, v in checkpoint.items():
            clean_key = None
            if '.decoder.' in k:
                clean_key = k.split('.decoder.')[1]
            elif k.startswith('decoder.'):
                clean_key = k.replace('decoder.', '', 1)
            elif has_module_prefix and k.startswith('module.'):
                clean_key = k.split('module.', 1)[1]
            if clean_key is not None:
                weights_to_load[clean_key] = v

    if not weights_to_load:
        logger.error('No decoder keys extracted from checkpoint.')
        return

    missing, unexpected = model.decoder.load_state_dict(weights_to_load, strict=False)
    logger.info('Loaded decoder: {} keys (missing {}, unexpected {}).'.format(
        len(weights_to_load) - len(missing), len(missing), len(unexpected)))

    model.eval()

    test_dataset = Getfile(base_dir=args.data_path, val_dir=target_test_dir, domain=1, num_classes=args.classes,
                           label_intensities=label_intensities, mode=target_mode, onehot=False, num_data=0, aug=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)
    logger.info('Test samples: {}'.format(len(test_dataset)))

    start_time = time.time()
    result = evaluate_full(test_dataloader, model, num_classes=args.classes)
    elapsed = str(timedelta(seconds=time.time() - start_time))

    summary = {
        'mode': args.mode,
        'source': source_mode,
        'target': target_mode,
        'class_names': None,
        'data_path': args.data_path,
        'test_dir': target_test_dir,
        'pth': args.pth,
        'num_classes': args.classes,
        'dino_size': args.dino_size,
        'gpu': args.gpu,
        'metric_notes': {
            'Dice': 'percent (x100)',
            'ASD': 'pixels',
            'HD95': 'pixels, GT must contain the class',
            'Jaccard': '0-1',
            'Sensitivity': '0-1, GT must contain the class',
            'Specificity': '0-1',
        },
        **result,
        'elapsed': elapsed,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    if args.json_out is None:
        args.json_out = os.path.join(
            os.path.dirname(os.path.abspath(args.pth)),
            'eval_{}_{}.json'.format(target_mode, datetime.now().strftime('%Y%m%d_%H%M%S')))
    with open(args.json_out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('--- Per-class metrics ---')
    for e in summary['per_class']:
        print('class {}: Dice={} ASD={} HD95={} Jaccard={} Sens={} Spec={} (n_gt={}, n_pred={}, phantom={})'.format(
            e['class'], e['Dice'], e['ASD'], e['HD95'], e['Jaccard'],
            e['Sensitivity'], e['Specificity'], e['n_gt'], e['n_pred'], e['phantom_n']))
    print('--- Mean ---', summary['mean'])
    print('pure_bg: total={}, perfect={}, with_pred={}; samples={}'.format(
        summary['n_pure_bg'], summary['n_pure_bg_perfect'], summary['n_pure_bg_with_pred'],
        summary['n_samples']))
    logger.info('JSON written to {} (took {})'.format(args.json_out, elapsed))


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main()