import numpy as np
import scipy
import torch
import torchvision
import torch
import torch.nn.functional as F


def _torch_cast_rgb(mat, cmap: str = 'inferno'):
    """把 [B,1,H,W] 的 [0,1] 单通道图经 matplotlib 色带映射为 [B,3,H,W] RGB（float [0,1]）。"""
    import matplotlib.cm as mcm
    cmap_obj = mcm.get_cmap(cmap) if hasattr(mcm, 'get_cmap') else __import__('matplotlib').colormaps[cmap]
    lut = torch.from_numpy(cmap_obj(np.linspace(0, 1, 256))[..., :3]).float().to(mat.device)
    idx = (mat.clamp(0, 1) * 255.0).round().long().squeeze(1)  # [B,H,W]
    rgb = lut[idx]  # [B,H,W,3]
    return rgb.permute(0, 3, 1, 2).contiguous()


def saliency_map(feat_map: torch.Tensor):
    """特征能量显著性图 [B,C,H,W] → [B,1,H,W]（L2 能量聚合 + 0.5%~99.5% 百分位裁剪）。"""
    feat_map = feat_map.detach().to('cpu').float()
    if feat_map.dim() != 4:
        raise ValueError(f"saliency_map expects [B,C,H,W], got {tuple(feat_map.shape)}")
    sal = feat_map.pow(2).sum(dim=1, keepdim=True).sqrt()  # [B,1,H,W]
    B = sal.shape[0]
    flat = sal.flatten(1)  # [B, HW]
    lo = torch.quantile(flat, 0.005, dim=1, keepdim=True).clamp_min(0.0)
    hi = torch.quantile(flat, 0.995, dim=1, keepdim=True)
    sal = (sal - lo.unsqueeze(-1).unsqueeze(-1)) / (hi.unsqueeze(-1).unsqueeze(-1) - lo.unsqueeze(-1).unsqueeze(-1) + 1e-6)
    return sal.clamp(0, 1)


def apply_color_map(map_2d: torch.Tensor, cmap: str = 'inferno', target_spatial=(256, 256)):
    """[B,1,H,W]（或 [B,H,W]）→ jet/inferno 彩色热力图，上采样到 target_spatial。"""
    if map_2d.dim() == 3:
        map_2d = map_2d.unsqueeze(1)
    rgb = _torch_cast_rgb(map_2d.clamp(0, 1).detach().to('cpu'), cmap=cmap)  # [B,3,H,W] on cpu
    rgb = F.interpolate(rgb, size=target_spatial, mode='bilinear', align_corners=False)
    return rgb.clamp(0, 1)


def overlay_on_image(base_img: torch.Tensor, heat_rgb: torch.Tensor, alpha: float = 0.5):
    """把热力图 RGB 半透明叠到原图（均 [0,1]，float，make_grid 兼容）。"""
    base = base_img.detach().to('cpu').float().clamp(0, 1)
    heat = heat_rgb.detach().to('cpu').float().clamp(0, 1)
    squeeze_out = False
    if base.dim() == 3:
        base = base.unsqueeze(0); squeeze_out = True
    if heat.dim() == 3:
        heat = heat.unsqueeze(0)
    if base.shape[-2:] != heat.shape[-2:]:
        base = F.interpolate(base, size=heat.shape[-2:], mode='bilinear', align_corners=False)
    out = (alpha * heat + (1.0 - alpha) * base).clamp(0, 1)
    return out[0] if squeeze_out else out


def partition_to_rgb(mask: torch.Tensor, target_spatial=(256, 256), color_pure=(0.0, 0.8, 0.0),
                     color_impure=(1.0, 0.2, 0.2)):
    """纯/不纯分区掩码 [B,1,Hp,Wp]（1=纯核）→ RGB：纯=绿，不纯=红，空 token 置黑。"""
    mask = mask.detach().to('cpu').float().clamp(0, 1)
    pure = mask[:, 0]  # [B,Hp,Wp]
    cp = torch.tensor(color_pure).view(1, 3, 1, 1)
    ci = torch.tensor(color_impure).view(1, 3, 1, 1)
    rgb = pure.unsqueeze(1) * cp + (1 - pure.unsqueeze(1)) * ci
    rgb = rgb.expand(-1, 3, -1, -1).contiguous()
    valid = pure.unsqueeze(1).expand(-1, 3, -1, -1)
    rgb = rgb * valid
    rgb = F.interpolate(rgb, size=target_spatial, mode='nearest')
    return rgb.clamp(0, 1)


def get_custom_bright_palette():
    palette = np.array([
        [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
    ], dtype=np.uint8)
    return torch.from_numpy(palette)


def denormalize_for_vis(img_tensor: torch.Tensor):
    device = torch.device("cpu")
    img_tensor_cpu = img_tensor.to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    return (img_tensor_cpu * std + mean).clamp(0, 1)


def colorize_mask(mask: torch.Tensor, num_classes: int, ignore_index: int = -1):
    mask_cpu = mask.to(torch.device("cpu"))

    if mask_cpu.dim() == 2:
        mask_cpu = mask_cpu.long()
        palette = get_custom_bright_palette()[:num_classes].to(mask_cpu.device)
        colored_mask = torch.zeros(mask_cpu.shape[0], mask_cpu.shape[1], 3, dtype=torch.uint8, device=mask_cpu.device)

        valid_mask = (mask_cpu != ignore_index)
        valid_indices = mask_cpu[valid_mask]

        if valid_indices.numel() > 0:
            valid_indices = valid_indices.clamp(max=num_classes - 1)
            colored_mask[valid_mask] = palette[valid_indices]

        colored_mask[~valid_mask] = 128

        return colored_mask.permute(2, 0, 1).float() / 255.0

    elif mask_cpu.dim() == 3 and torch.is_floating_point(mask_cpu):
        palette = get_custom_bright_palette()[:num_classes].to(mask_cpu.device).float()
        valid_mask = (mask_cpu[0, :, :] != ignore_index)
        valid_probs = mask_cpu * valid_mask.unsqueeze(0).float()
        colored_mask = torch.matmul(valid_probs.permute(1, 2, 0), palette)
        colored_mask[~valid_mask] = 128
        return colored_mask.clamp(0, 255).permute(2, 0, 1).float() / 255.0

    else:
        raise ValueError(f"Unsupported mask format: shape={mask_cpu.shape}, dtype={mask_cpu.dtype}.")


def log_visualizations(writer, global_step, batch_data, num_classes, args):
    """
    统一 TensorBoard 看板命名（英文，对齐论文术语）：
      Viz/01_Source_Decoder (Input, GT, Pred, HFM_Pred)
      Viz/02_Target_Decoder (Input, SD_Pred, EMA_Pred, PseudoLabel)
      Viz/03_Source_Features (Encoder, F_s_cross, F_s2t, Attention)
      Viz/04_Target_Features (Encoder, F_t_cross, F_t2s, Attention)
      Viz/05_Source_HFM_Branch (Unfold, Purity, Partition, Modulated, Fold_Fs_cross)
      Viz/06_Target_HFM_Branch (Unfold, Purity, Partition, Modulated, Fold_Ft_cross)
    """
    idx_to_show = 0
    ignore_index = getattr(args, 'ignore_index', -1)
    spatial = (256, 256)

    def denorm_raw(key):
        t = batch_data.get(key)
        if t is None or t.shape[0] <= idx_to_show:
            return None
        return denormalize_for_vis(t[idx_to_show])

    def heat_overlay(base_key, feat_key, need_saliency=False, cmap='inferno', alpha=0.5):
        base = denorm_raw(base_key)
        feat = batch_data.get(feat_key)
        if base is None or feat is None:
            return None
        if need_saliency:
            heat = apply_color_map(saliency_map(feat), cmap=cmap, target_spatial=spatial)[0]
        else:
            if feat.dim() == 3:
                feat = feat.unsqueeze(1)
            heat = apply_color_map(feat, cmap=cmap, target_spatial=spatial)[0]
        return overlay_on_image(base, heat, alpha=alpha)

    def partition_overlay(base_key, mask_key, alpha=0.35):
        base = denorm_raw(base_key)
        mask = batch_data.get(mask_key)
        if base is None or mask is None:
            return None
        rgb = partition_to_rgb(mask, target_spatial=spatial)[0]
        return overlay_on_image(base, rgb, alpha=alpha)

    def modulated_overlay(base_key, mask_key, alpha=0.4):
        base = denorm_raw(base_key)
        mask = batch_data.get(mask_key)
        if base is None or mask is None:
            return None
        hot = apply_color_map(mask, cmap='inferno', target_spatial=spatial)[0]
        return overlay_on_image(base, hot, alpha=alpha)

    # --- 1. Viz/01 源域解码看板 ---
    source_img, source_lbl = batch_data.get('source_img'), batch_data.get('source_lbl')
    s_ori_logits, s_cross_logits = batch_data.get('s_logits_ori'), batch_data.get('s_logits_cross')
    if all(t is not None for t in [source_img, source_lbl, s_ori_logits, s_cross_logits]):
        if source_img.shape[0] > idx_to_show:
            grid_source = torchvision.utils.make_grid(
                [denormalize_for_vis(source_img[idx_to_show]),
                 colorize_mask(source_lbl[idx_to_show], num_classes, ignore_index),
                 colorize_mask(torch.argmax(s_ori_logits[idx_to_show], dim=0), num_classes, ignore_index),
                 colorize_mask(torch.argmax(s_cross_logits[idx_to_show], dim=0), num_classes, ignore_index)],
                nrow=4
            )
            writer.add_image('Viz/01_Source_Decoder (Input, GT, Pred, HFM_Pred)', grid_source, global_step)

    # --- 2. Viz/02 目标域解码看板 ---
    target_img = batch_data.get('target_img')
    target_logits_student = batch_data.get('target_logits_ori_student')
    target_logits_ema = batch_data.get('target_logits_ema')
    pseudo_labels_hard = batch_data.get('pseudo_labels')
    if all(t is not None for t in [target_img, target_logits_student, target_logits_ema, pseudo_labels_hard]):
        if target_img.shape[0] > idx_to_show:
            grid_target = torchvision.utils.make_grid(
                [denormalize_for_vis(target_img[idx_to_show]),
                 colorize_mask(torch.argmax(target_logits_student[idx_to_show], dim=0), num_classes, ignore_index),
                 colorize_mask(torch.argmax(target_logits_ema[idx_to_show], dim=0), num_classes, ignore_index),
                 colorize_mask(pseudo_labels_hard[idx_to_show], num_classes, ignore_index)],
                nrow=4
            )
            writer.add_image('Viz/02_Target_Decoder (Input, SD_Pred, EMA_Pred, PseudoLabel)', grid_target, global_step)

    # --- 3. Viz/03 源域特征看板（Encoder / F_s,cross / F_s→t / Attention，inferno 叠加原图）---
    def render_feat_board(board_tag, base_key, feat_keys):
        cols = []
        for fk, need_sal in feat_keys:
            c = heat_overlay(base_key, fk, need_saliency=need_sal)
            if c is not None:
                cols.append(c)
        if cols:
            grid = torchvision.utils.make_grid(cols, nrow=len(cols), padding=4, pad_value=0.5)
            writer.add_image(board_tag, grid, global_step)

    render_feat_board('Viz/03_Source_Features (Encoder, F_s_cross, F_s2t, Attention)',
                      'source_img',
                      [('s_feat_map', True), ('s_feat_cross', True), ('s_feat_stylized', True), ('s_attn_map', False)])
    render_feat_board('Viz/04_Target_Features (Encoder, F_t_cross, F_t2s, Attention)',
                      'target_img',
                      [('t_feat_map', True), ('t_feat_cross', True), ('t_feat_stylized', True), ('t_attn_map', False)])

    # --- 4. Viz/05/06 HFM unfold→fold 链路看板 ---
    # 五列：Unfold 细网格特征 / Purity 纯度 / Partition 纯-不纯 / Modulated 调制位 / Fold 折叠结果
    def render_hfm_branch_board(board_tag, base_key, suffix):
        cols = []
        c = heat_overlay(base_key, f'{suffix}_feat_fine', need_saliency=True)
        if c is not None:
            cols.append(c)
        c = heat_overlay(base_key, f'{suffix}_purity_maps', need_saliency=False)
        if c is not None:
            cols.append(c)
        c = partition_overlay(base_key, f'{suffix}_partition_masks')
        if c is not None:
            cols.append(c)
        c = modulated_overlay(base_key, f'{suffix}_modulated_masks')
        if c is not None:
            cols.append(c)
        c = heat_overlay(base_key, f'{suffix}_cross_fine', need_saliency=True)
        if c is not None:
            cols.append(c)
        if len(cols) == 5:
            grid = torchvision.utils.make_grid(cols, nrow=5, padding=4, pad_value=0.5)
            writer.add_image(board_tag, grid, global_step)

    render_hfm_branch_board('Viz/05_Source_HFM_Branch (Unfold, Purity, Partition, Modulated, Fold_Fs_cross)',
                            'source_img', 's')
    render_hfm_branch_board('Viz/06_Target_HFM_Branch (Unfold, Purity, Partition, Modulated, Fold_Ft_cross)',
                            'target_img', 't')

    # --- 5. Viz/07/08 DINOv3 上采样特征 + Pure patch 结果看板 ---
    # 三列：Upsampled_DINOv3_Feat（双线性上采样到目标分辨率后的特征能量）/ Purity 纯度 / Partition 纯-不纯
    def dino_upsampled(feat_key, cmap='viridis'):
        feat = batch_data.get(feat_key)
        if feat is None or feat.shape[0] <= idx_to_show:
            return None
        one = feat[idx_to_show:idx_to_show + 1].detach().float().to('cpu')
        up = F.interpolate(one, size=spatial, mode='bilinear', align_corners=False)
        return apply_color_map(saliency_map(up), cmap=cmap, target_spatial=spatial)[0]

    def render_dino_pure_board(board_tag, base_key, suffix):
        cols = []
        c = dino_upsampled(f'{suffix}_feat_map')
        if c is not None:
            cols.append(c)
        c = heat_overlay(base_key, f'{suffix}_purity_maps', need_saliency=False, cmap='viridis')
        if c is not None:
            cols.append(c)
        c = partition_overlay(base_key, f'{suffix}_partition_masks')
        if c is not None:
            cols.append(c)
        if len(cols) == 3:
            grid = torchvision.utils.make_grid(cols, nrow=3, padding=4, pad_value=0.5)
            writer.add_image(board_tag, grid, global_step)

    render_dino_pure_board('Viz/07_Source_DINOv3 (Upsampled_Feat, Purity, Partition)', 'source_img', 's')
    render_dino_pure_board('Viz/08_Target_DINOv3 (Upsampled_Feat, Purity, Partition)', 'target_img', 't')

    # --- 6. Viz/09 (HPE) / Viz/10 (SAP) 类级矩阵热力图看板 ---
    # HPE：每类结构相似度/布局相似度/融合后 soft-min 分数（batch 样本）
    # SAP：跨视图异常分数 theta 阈值线/异常类标记/各视图像素占比
    render_hpe_sap_boards(writer, global_step, batch_data, num_classes)

def render_hpe_sap_boards(writer, global_step, batch_data, num_classes):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return

    n_samples = batch_data.get('pl_hpe_final_scores')
    if n_samples is not None and n_samples.numel() > 0:
        shape_sim = batch_data.get('pl_hpe_shape_sim')
        layout_sim = batch_data.get('pl_hpe_layout_sim')
        intra = batch_data.get('pl_hpe_intra_scores')
        inter = batch_data.get('pl_hpe_inter_scores')
        final = batch_data.get('pl_hpe_final_scores')
        if all(t is not None for t in [shape_sim, layout_sim, intra, inter, final]):
            B = shape_sim.shape[0]
            C = shape_sim.shape[1]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
            ax = axes[0]
            im = ax.imshow(shape_sim.cpu().numpy().T, cmap='jet', aspect='auto')
            ax.set_title('Shape similarity\n(per-class, isoperimetric)')
            ax.set_xlabel('batch sample')
            ax.set_yticks(range(C)); ax.set_yticklabels([f'c{c}' for c in range(C)])
            fig.colorbar(im, ax=ax)

            ax = axes[1]
            im = ax.imshow(layout_sim.cpu().mean(dim=0).numpy(), cmap='jet', aspect='auto')
            ax.set_title('Layout similarity\n(mean over batch, c1→c2)')
            ax.set_xlabel('c2'); ax.set_ylabel('c1')
            fig.colorbar(im, ax=ax)

            ax = axes[2]
            x = torch.arange(B).cpu().numpy()
            ax.plot(x, intra.cpu().numpy(), 'o-', label='S_intra', markersize=3)
            ax.plot(x, inter.cpu().numpy(), 's-', label='S_inter', markersize=3)
            ax.plot(x, final.cpu().numpy(), '^-', label='S_final', markersize=3)
            ax.set_title('Soft-min fused scores (HPE)')
            ax.set_xlabel('batch sample'); ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            writer.add_image('Viz/09_HPE (Shape, Layout, Fused Softmin)', _fig_to_tensor(fig), global_step)

    # SAP：跨视图异常分数矩阵 + theta 阈值 + 异常类与各视图像素占比
    anomaly_scores = batch_data.get('pl_sap_anomaly_scores')
    if anomaly_scores is not None and anomaly_scores.numel() > 0:
        is_anom = batch_data.get('pl_sap_is_anomalous_class')
        counts = batch_data.get('pl_sap_pixel_counts_per_view')
        theta = batch_data.get('pl_sap_anomaly_threshold')
        C = anomaly_scores.shape[1]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        ax = axes[0]
        scores_np = anomaly_scores.cpu().numpy()
        im = ax.imshow(scores_np.T, cmap='inferno', aspect='auto')
        ax.set_title(f'Cross-view anomaly scores (std/mean)\nθ={float(theta):.3f}' if theta is not None
                     else 'Cross-view anomaly scores (std/mean)')
        ax.set_xlabel('batch sample')
        ax.set_yticks(range(C)); ax.set_yticklabels([f'c{c}' for c in range(C)])
        fig.colorbar(im, ax=ax)

        ax = axes[1]
        if is_anom is not None:
            im = ax.imshow(is_anom.cpu().numpy().T.astype(float), cmap='Reds', aspect='auto', vmin=0, vmax=1)
            ax.set_title('Anomalous class (masked to ignore)')
            ax.set_xlabel('batch sample')
            ax.set_yticks(range(C)); ax.set_yticklabels([f'c{c}' for c in range(C)])
            fig.colorbar(im, ax=ax)

        ax = axes[2]
        if counts is not None:
            ratio = counts / counts.sum(dim=1, keepdim=True).clamp(min=1.0)
            im = ax.imshow(ratio.cpu().mean(0).numpy().T, cmap='viridis', aspect='auto')
            ax.set_title('Mean per-class pixel ratio\n(across teacher views)')
            ax.set_xlabel('view'); ax.set_ylabel('class')
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        writer.add_image('Viz/10_SAP (Anomaly, Theta, Pixel_Ratio)', _fig_to_tensor(fig), global_step)


def _fig_to_tensor(fig):
    import matplotlib.pyplot as plt
    try:
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    finally:
        plt.close(fig)
    return torch.from_numpy(buf).permute(2, 0, 1).float() / 255.0

def reverse_mode(mode):
    if mode == "CT":
        return 'MR'
    elif mode == "MR":
        return 'CT'
    elif mode == "ABCT":
        return 'ABMR'
    elif mode == "ABMR":
        return 'ABCT'
    else:
        return 'ABCT'