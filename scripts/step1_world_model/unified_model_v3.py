#!/usr/bin/env python3
"""Core components for the cross-species blood-to-brain world model."""
import sys, os, json, time, argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROJECT = Path(os.environ.get("AD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(PROJECT))

from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from dynamics import DriftNet, integrate_ode
except ImportError:
    # fallback: 内联 DriftNet(如果 dynamics.py 不在 path)
    from dynamics import DriftNet, integrate_ode


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

@dataclass
class Config:
    latent_dim: int = 64
    esm_dim: int = 640
    enc_hidden: int = 256
    enc_depth: int = 2
    ode_hidden: int = 128
    ode_depth: int = 2
    ode_time_freq: int = 4
    ode_n_steps: int = 4
    dec_z_hidden: int = 256      # z→esm 解码器
    dec_z_depth: int = 2
    epochs: int = 200
    steps_per_epoch: int = 200
    batch_size: int = 128
    lr: float = 2e-4
    beta_kl: float = 0.01
    beta_cos: float = 1.0        # cosine loss 权重
    beta_mse: float = 2.0        # mse loss 权重(基因空间; v3.3 提高到 2.0 —— 原 0.5 时跨组织任务被 recon 压制, 输出幅度缩水, mse 输给条件常数基线)
    beta_recon: float = 0.1      # 自编码重建权重(v3.3 降到 0.1 —— 原硬编码 0.5 时 recon 主导梯度)
    zero_weight: float = 20.0    # 非零基因的 MSE 权重(防坍缩核心)
    center_genes: bool = True    # per-gene 中心化(抽掉均值预测器的免费午餐, 抗坍缩)
    beta_drift: float = 1e-2     # ODE 位移惩罚(防 drift_net 爆炸; 正常 z 量级下可忽略)
    beta_rev: float = 1.0        # v6: 纯 reverse 训练(血→脑),forward 路线已删除
    separate_rev_drift: bool = True  # v6: 独立反向漂移头(血→脑 直接学习的场)
    paired_by_genotype: bool = True  # 配对采样(v3.3+): 脑配对键 k → 同 k 血靶; 无标签自动回退随机
    use_ct_cond: bool = True   # 细胞类型条件 ODE(v4): cond = onehot(脑细胞类型) ++ onehot(血细胞类型)
    cond_dim: int = 0          # 条件向量维度(训练时按标签数自动填充)
    ct_brain_labels: list = None  # 训练时填充, 供 v-KO 组合条件复用
    ct_blood_labels: list = None
    compartment_pair: bool = True  # 区室限定配对(v4.1): 小胶质(Micro/mg)仅配同键髓系腔室, 防无关细胞稀释信号
    myeloid_compartment: tuple = ("MyeloidP", "Mono", "HSPC", "pDC")  # 髓系腔室(血端标签)
    micro_labels: tuple = ("Micro", "mg")  # 小胶质(脑侧标签, 鼠/人)
    output_dir: str = "results/step1_world_model_v3"
    model_dir: str = "models/world_model_v3"
    seed: int = 42                 # 全局种子(bit 级可复现, 2026-08-09 拍板): 模型初始化/dataloader 全固定


# ═══════════════════════════════════════════════════════════════
# 跨物种细胞类型对齐 (v5 joint): (species, 原标签) → 对齐标签
# 谱系一致的保守映射; HSPC/MyeloidP/Pericyte 为鼠独有(人端 PBMC 无对应)。
# ═══════════════════════════════════════════════════════════════
BRAIN_ALIGN = ["Micro", "Oligo", "Astro", "OPC", "Neuron", "Endo", "Pericyte"]
# v5.2: 人血 CD4T/CD8T 细分(Leiden 簇级标签); 鼠血 T 标记太低保真不拆, 保留 T 维
BLOOD_ALIGN = ["HSPC", "MyeloidP", "Mono", "B", "T", "CD4T", "CD8T", "NK", "DC", "Mega", "Ery"]

CT_ALIGN = {
    ("mouse", "Micro"): "Micro", ("mouse", "Oligo"): "Oligo", ("mouse", "Astro"): "Astro",
    ("mouse", "OPC"): "OPC", ("mouse", "Neuron"): "Neuron", ("mouse", "Endo"): "Endo",
    ("mouse", "Pericyte"): "Pericyte",
    ("human", "mg"): "Micro", ("human", "oligo"): "Oligo", ("human", "astro"): "Astro",
    ("human", "OPC"): "OPC", ("human", "neuron"): "Neuron", ("human", "endo"): "Endo",
    ("mouse", "HSPC"): "HSPC", ("mouse", "MyeloidP"): "MyeloidP", ("mouse", "Mono"): "Mono",
    ("mouse", "B"): "B", ("mouse", "T"): "T",  # 鼠 T 保留 T 维(标记太低无法保真拆分)
    ("mouse", "CD4T"): "CD4T", ("mouse", "CD8T"): "CD8T",
    ("mouse", "NK"): "NK",
    ("mouse", "pDC"): "DC", ("mouse", "Mega"): "Mega", ("mouse", "Ery"): "Ery",
    ("human", "Mono"): "Mono", ("human", "CD14_Mono"): "Mono",
    ("human", "B"): "B",
    ("human", "CD4T"): "CD4T", ("human", "CD8T"): "CD8T",  # v5.2: Leiden 细分
    ("human", "NK"): "NK", ("human", "DC"): "DC",
    ("human", "Mega"): "Mega", ("human", "Ery"): "Ery",
    # human Prolif/Unresolved 无映射 → 自动 NA
}


def align_ct(labels, species):
    """原细胞类型标签 → 跨物种对齐标签 (无法对齐的标 'NA', 进不了对齐 onehot)。"""
    return np.array([CT_ALIGN.get((species, str(l)), "NA") for l in labels])

# ═══════════════════════════════════════════════════════════════
# 模型组件
# ═══════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """ESM2 空间(640) → latent z(64)。

    注意: 不用 LayerNorm —— 全链路无 LN(v3.2)。LN 的尺度不变性会吸收
    ODE 爆炸的 z 并抹掉 x_esm 的输入尺度信息, 详见 DecoderZ 注释。
    """
    def __init__(self, esm_dim=640, latent_dim=64, hidden=256, depth=2, dropout=0.1):
        super().__init__()
        layers = []
        d = esm_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(d, latent_dim)
        self.fc_logvar = nn.Linear(d, latent_dim)

    def forward(self, x):
        h = self.backbone(x)
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), -10, 10)

    @staticmethod
    def reparam(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


class DecoderZ(nn.Module):
    """latent z(64) → ESM2 空间(640)。

    注意: 不用 LayerNorm —— LN 的尺度不变性会吸收 ODE 爆炸的 z
    (旧版 z_blood 爆到 5e5 仍被 LN 归一化成常数输出, 造成后验坍缩假象)。
    """
    def __init__(self, latent_dim=64, esm_dim=640, hidden=256, depth=2, dropout=0.1):
        super().__init__()
        layers = []
        d = latent_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(d, esm_dim)

    def forward(self, z):
        return self.out(self.backbone(z))


class WorldModelV3(nn.Module):
    """半训练 Decoder 架构。

    数据流:
      x_gene[11982] → @ESM2(固定) → x_esm[640]
      → Encoder → z[64] → ODE → z_blood[64]
      → DecoderZ → x_esm_pred[640]
      → DecOut(640→11982, 可训练) → x_gene_pred[11982]
      Loss 在 11982 基因空间算。

    边提取: 世界模型虚拟敲除(v-KO, 逐源基因置零读血端 Δ)。
    不需要 pinv(ESM2), 无投影伪影。
    """
    def __init__(self, n_genes, esm_matrix, config):
        super().__init__()
        self.n_genes = n_genes
        self.esm_dim = esm_matrix.shape[1]
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ESM2 矩阵(固定, 不训练)
        self.register_buffer("esm_matrix", torch.from_numpy(esm_matrix).float())
        # per-gene 均值(center_genes 启用时在 train() 里设置; 默认 0 = 不中心化, 向后兼容)
        self.register_buffer("gene_mean", torch.zeros(n_genes))

        # 可训练组件
        self.encoder = Encoder(self.esm_dim, config.latent_dim, config.enc_hidden, config.enc_depth)
        self.drift_net = DriftNet(config.latent_dim, config.ode_hidden, config.ode_depth,
                                  config.ode_time_freq, cond_dim=config.cond_dim)
        # v5.2: 独立反向漂移头 (血→脑 = 直接学习的向量场, 而非 forward 场的数学反演)
        # 动因: v5.1 反向损失下 reverse 边 decoy minFDR 仍=100% (人/鼠均然),
        #       反演的 KO 灵敏度由共享编码器几何主导, 与配对无关 → 反向必须有自己的场
        self.drift_net_rev = DriftNet(config.latent_dim, config.ode_hidden, config.ode_depth,
                                      config.ode_time_freq, cond_dim=config.cond_dim) \
            if getattr(config, "separate_rev_drift", False) else None
        self.decoder_z = DecoderZ(config.latent_dim, self.esm_dim, config.dec_z_hidden, config.dec_z_depth)

        # 半训练 Decoder: 640→11982, 初始化为 ESM2^T
        # 这个层学习"ESM2空间→基因空间"的逆映射, 替代 pinv
        self.dec_out = nn.Linear(self.esm_dim, n_genes, bias=False)
        with torch.no_grad():
            self.dec_out.weight.data.copy_(torch.from_numpy(esm_matrix).float())  # W=(n_genes,640), y=x@W.T 等价 x@ESM2^T
        log(f"  Dec_out 初始化: ESM2^T ({n_genes}×{self.esm_dim}), 可训练")

    def forward_ode(self, z, direction="brain_to_blood", cond=None):
        n_steps = self.config.ode_n_steps
        device = z.device
        if direction == "brain_to_blood":
            t0, t1 = torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
            net = self.drift_net
        elif self.drift_net_rev is not None:
            # v5.2: 独立反向场, blood→brain 正向积分 (t: 0→1)
            t0, t1 = torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
            net = self.drift_net_rev
        else:
            # 旧行为: forward 场反向积分 (t: 1→0, 数学反演)
            t0, t1 = torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)
            net = self.drift_net
        # 条件 ODE: cond 通过闭包注入, integrate_ode 无需感知 (dynamics.py 既定模式)
        f = (lambda x, t: net(x, t, cond)) \
            if (cond is not None and self.config.cond_dim > 0) else net
        return integrate_ode(f, z, t0, t1, n_steps, method="rk4")

    def _integrate(self, z, t0, t1, n_steps=4, cond=None):
        device = z.device
        f = (lambda x, t: self.drift_net(x, t, cond)) \
            if (cond is not None and self.config.cond_dim > 0) else self.drift_net
        return integrate_ode(f, z,
                             torch.tensor(float(t0), device=device),
                             torch.tensor(float(t1), device=device), n_steps)

    def kl_loss(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, x_brain_gene, cond=None):
        """完整前向: 基因空间输入 → ESM2投影 → encode → ODE → decode → 基因空间输出。

        返回 (x_blood_pred_gene, x_brain_recon_gene, mu, logvar, z_blood)。
        x_blood_pred_gene 是基因空间的预测血, 直接和真实血算 loss。
        z_blood 额外返回用于 ODE 位移惩罚(防 drift 爆炸)。
        cond: 细胞类型条件 (B, cond_dim), v4 起传入条件 ODE。
        """
        # per-gene 中心化(常数平移在 KO 差分中抵消, 同时抽掉均值预测器的免费午餐)
        x_brain_c = x_brain_gene - self.gene_mean
        # 基因→ESM2
        x_brain_esm = x_brain_c @ self.esm_matrix  # (B, 640)

        # Encode
        mu, logvar = self.encoder(x_brain_esm)
        z_brain = Encoder.reparam(mu, logvar)

        # ODE: brain → blood (条件化)
        z_blood = self.forward_ode(z_brain, "brain_to_blood", cond)

        # Decode: z → esm → gene
        x_blood_esm_pred = self.decoder_z(z_blood)  # (B, 640)
        x_blood_gene_pred = self.dec_out(x_blood_esm_pred)  # (B, 11982)

        # 自编码重建(防坍缩)
        x_brain_esm_recon = self.decoder_z(z_brain)
        x_brain_gene_recon = self.dec_out(x_brain_esm_recon)

        return x_blood_gene_pred, x_brain_gene_recon, mu, logvar, z_blood

    def compute_loss(self, x_brain_gene, x_blood_gene, cond=None):
        """Zero-Masked MSE + Cosine + KL(在基因空间算)。

        Zero-Masked MSE: 非零基因权重=zero_weight, 零基因权重=1。
        防止 decoder 输出全零/常数(坍缩)。
        cond: 细胞类型条件 (B, cond_dim), v4 起。
        """
        cfg = self.config
        x_pred, x_recon, mu, logvar, z_blood = self.forward(x_brain_gene, cond)
        # 中心化目标(mask 仍用原始未中心化值定义"非零基因")
        x_blood_c = x_blood_gene - self.gene_mean
        x_brain_c = x_brain_gene - self.gene_mean

        # Zero-Masked MSE(预测血 vs 真实血, 中心化空间)
        nonzero_mask = (x_blood_gene.abs() > 1e-6).float()  # 非零基因
        weight = nonzero_mask * cfg.zero_weight + (1 - nonzero_mask) * 1.0
        mse_loss = ((x_pred - x_blood_c) ** 2 * weight).mean()

        # Cosine loss(中心化后 ≈ 相关, 不再被共享均值项 trivially 满足)
        cos_loss = 1.0 - F.cosine_similarity(x_pred, x_blood_c, dim=-1).mean()

        # 自编码重建(同样 Zero-Masked)
        nonzero_brain = (x_brain_gene.abs() > 1e-6).float()
        weight_brain = nonzero_brain * cfg.zero_weight + (1 - nonzero_brain) * 1.0
        recon_loss = ((x_recon - x_brain_c) ** 2 * weight_brain).mean()

        # KL
        kl = self.kl_loss(mu, logvar)

        # ODE 位移惩罚: 防 drift_net 爆炸(旧版 z_blood 曾爆到 5e5)
        drift_reg = (z_blood - mu).pow(2).mean()

        loss = (cfg.beta_mse * mse_loss + cfg.beta_cos * cos_loss
                + cfg.beta_recon * recon_loss + cfg.beta_kl * kl
                + cfg.beta_drift * drift_reg)
        out = {
            'loss': loss, 'mse': mse_loss, 'cos': cos_loss,
            'recon': recon_loss, 'kl': kl, 'drift': drift_reg,
            'nonzero_frac': nonzero_mask.mean().item(),
        }

        # v5.1 反向通路 (血→脑): 让 ODE 反向积分承载真实学习的映射。
        # 动因: forward-only 训练下, reverse 边 = ODE 纯数学反演, 诱饵库 FDR=100%
        # (零分布尾部不输真实模型); 加反向预测损失后, 反向才有监督信号。
        if cfg.beta_rev > 0:
            x_blood_esm = x_blood_c @ self.esm_matrix
            mu2, logvar2 = self.encoder(x_blood_esm)
            z_blood_in = Encoder.reparam(mu2, logvar2)
            z_brain_hat = self.forward_ode(z_blood_in, "blood_to_brain", cond)
            x_brain_hat = self.dec_out(self.decoder_z(z_brain_hat))
            # 目标=脑侧, mask 用脑侧非零基因 (weight_brain 已按原逻辑定义)
            mse_rev = ((x_brain_hat - x_brain_c) ** 2 * weight_brain).mean()
            cos_rev = 1.0 - F.cosine_similarity(x_brain_hat, x_brain_c, dim=-1).mean()
            kl2 = self.kl_loss(mu2, logvar2)
            drift_rev = (z_brain_hat - mu2).pow(2).mean()
            rev = (cfg.beta_mse * mse_rev + cfg.beta_cos * cos_rev
                   + cfg.beta_kl * kl2 + cfg.beta_drift * drift_rev)
            out['loss'] = loss + cfg.beta_rev * rev
            out['mse_rev'] = mse_rev
            out['cos_rev'] = cos_rev
        return out


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def build_esm2_matrix(gene_names, member_level=False):
    """构建 ESM2 投影矩阵(从缓存或在线算)。member_level 时用 OG 成员均值缓存。"""
    cache_name = "esm2_member_level_dim640.pt" if member_level \
        else "esm2_all_genes_t30_150M_dim640.pt"
    cache_path = PROJECT / "cache" / "esm2_embeddings" / cache_name
    if cache_path.exists():
        import torch
        cache = torch.load(cache_path, map_location='cpu', weights_only=False)
        actual_dim = len(next(iter(cache.values())))
    else:
        log("ESM2 缓存不存在, 用随机初始化(后续补算)")
        cache = {g.upper(): np.random.randn(640).astype(np.float32) * 0.1 for g in gene_names}
        actual_dim = 640

    n = len(gene_names)
    mat = np.zeros((n, actual_dim), dtype=np.float32)
    covered = 0
    for i, g in enumerate(gene_names):
        if g in cache:
            mat[i] = np.asarray(cache[g], dtype=np.float32)
            covered += 1
        elif g.upper() in cache:
            mat[i] = np.asarray(cache[g.upper()], dtype=np.float32)
            covered += 1
    log(f"ESM2 矩阵: {mat.shape}, 覆盖 {covered}/{n} ({covered/n:.1%})")

    # 列归一化
    col_norms = np.linalg.norm(mat, axis=0, keepdims=True)
    col_norms[col_norms < 1e-8] = 1.0
    return mat / col_norms


def load_data(pair_key="mouse", species="mouse", member_level=False):
    """加载脑 + 血端, 返回
    (X_brain, X_blood, gene_names, brain_key, blood_key, brain_ct, blood_ct,
     brain_donor, blood_donor)。

    species='mouse': 脑=5xFAD h5ad, 血端=BM12(或回退旧 Blood); pair_key mouse/genotype。
    species='human': 脑=GSE138852, 血=GSE181279; pair_key 失效, 固定疾病配对(AD/CTRL)。
    member_level=True: 读 data/member_level/ (OG 求和列空间 + 私有节点, build_member_level.py 产出)。
    """
    if member_level:
        return _load_member(pair_key, species)
    if species == "human":
        return _load_human()
    import scanpy as sc

    norm_path = PROJECT / "data/transcriptomics/5xFAD_expression_matrix_for_step4.h5ad"
    bm12_path = PROJECT / "data/transcriptomics/BM12_blood_genotype_labeled.h5ad"

    # 脑(含基因型与样本→鼠个体映射)
    log(f"加载脑数据: {norm_path.name}")
    adata = sc.read_h5ad(norm_path)
    X = adata.X
    if hasattr(X, 'toarray'): X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    gene_names = list(adata.var_names.astype(str))
    brain_mask = (adata.obs['tissue'].astype(str) == 'Brain').values
    X_brain = np.asarray(X[brain_mask], dtype=np.float32)
    brain_geno = adata.obs['genotype'].astype(str).values[brain_mask]
    SAMPLE2MOUSE = {'955': 'WT_1', '956': '5xFAD_1', '957': 'WT_2', '958': '5xFAD_2'}
    brain_mouse = pd.Series(adata.obs['sample'].astype(str).values[brain_mask]) \
        .map(SAMPLE2MOUSE).values
    log(f"  脑: {X_brain.shape}, 基因型: {dict(zip(*np.unique(brain_geno, return_counts=True)))}, "
        f"鼠个体: {dict(zip(*np.unique(brain_mouse, return_counts=True)))}")

    # 血端
    if bm12_path.exists():
        log(f"加载 BM1/BM2 血端(带基因型): {bm12_path.name}")
        b = sc.read_h5ad(bm12_path)
        Xb = b.X
        if hasattr(Xb, 'toarray'): Xb = Xb.toarray()
        X_blood = np.asarray(Xb, dtype=np.float32)
        assert list(b.var_names.astype(str)) == gene_names, "BM12 基因顺序与参考 h5ad 不一致!"
        blood_geno = b.obs['genotype'].astype(str).values
        # obs mouse 形如 5xFAD_1_femur / WT_2_skull → 鼠个体 5xFAD_1 / WT_2
        blood_mouse = pd.Series(b.obs['mouse'].astype(str).values) \
            .map(lambda s: s.rsplit('_', 1)[0]).values
        n12 = len(blood_geno)
        log(f"  血(BM12): {X_blood.shape}, 基因型: {dict(zip(*np.unique(blood_geno, return_counts=True)))}")

        # BM3/BM4 髓系+HSPC (demux 后带标签): 与 BM12 合并, 扩大配对池
        bm34_path = PROJECT / "data/transcriptomics/BM34_blood_genotype_labeled.h5ad"
        blood_ct34 = None
        if bm34_path.exists():
            b34 = sc.read_h5ad(bm34_path)
            Xc = b34.X
            if hasattr(Xc, 'toarray'): Xc = Xc.toarray()
            Xc = np.asarray(Xc, dtype=np.float32)
            assert list(b34.var_names.astype(str)) == gene_names, "BM34 基因顺序不一致!"
            X_blood = np.vstack([X_blood, Xc])
            blood_geno = np.concatenate([blood_geno, b34.obs['genotype'].astype(str).values])
            blood_mouse = np.concatenate([blood_mouse, b34.obs['mouse'].astype(str).values])
            blood_ct34 = b34.obs['celltype'].astype(str).values
            log(f"  血(BM34 髓系): +{Xc.shape[0]} 细胞, 合并后 {X_blood.shape[0]}")
        log(f"  血(合并): 基因型: {dict(zip(*np.unique(blood_geno, return_counts=True)))}, "
            f"鼠个体: {dict(zip(*np.unique(blood_mouse, return_counts=True)))}")
    else:
        log("BM12 血端不存在, 回退 5xFAD h5ad Blood (mixed, 无标签)")
        blood_mask = (adata.obs['tissue'].astype(str) == 'Blood').values
        X_blood = np.asarray(X[blood_mask], dtype=np.float32)
        blood_geno = None
        blood_mouse = None
        blood_ct34 = None
        log(f"  血: {X_blood.shape}")

    if pair_key == "mouse":
        log(f"配对键: 鼠个体 (个体配对)")
        brain_key, blood_key = brain_mouse, blood_mouse
    else:
        log(f"配对键: 基因型")
        brain_key, blood_key = brain_geno, blood_geno

    # 细胞类型标注 (v4): build_celltypes.py 产出, 按 obs_names 对齐
    brain_ct = blood_ct = None
    brain_ct_path = PROJECT / "data/transcriptomics/brain_celltypes.csv"
    if brain_ct_path.exists():
        ctbl = pd.read_csv(brain_ct_path, index_col=0)["celltype"]
        brain_ct = ctbl.reindex(adata.obs_names[brain_mask]).astype(str).values
        n_nan = (brain_ct == 'nan').sum()
        log(f"  脑细胞类型: {dict(zip(*np.unique(brain_ct, return_counts=True)))}" +
            (f" [警告 {n_nan} 未匹配]" if n_nan else ""))
    if blood_geno is not None:
        blood_ct_path = PROJECT / "data/transcriptomics/bm12_celltypes.csv"
        if blood_ct_path.exists():
            ctbl = pd.read_csv(blood_ct_path, index_col=0)["celltype"]
            blood_ct = ctbl.reindex(b.obs_names).astype(str).values
            if blood_ct34 is not None:
                blood_ct = np.concatenate([blood_ct, blood_ct34])
            n_nan = (blood_ct == 'nan').sum()
            log(f"  血细胞类型: {dict(zip(*np.unique(blood_ct, return_counts=True)))}" +
                (f" [警告 {n_nan} 未匹配]" if n_nan else ""))
        elif blood_ct34 is not None:
            blood_ct = blood_ct34

    return X_brain, X_blood, gene_names, brain_key, blood_key, brain_ct, blood_ct, \
        brain_mouse, blood_mouse


def _load_human():
    """人侧数据: 脑 GSE138852 + 血 GSE181279 (build_human_data.py 产出)。

    配对键 = 疾病标签 (AD/CTRL); 供体 = donor_pair(脑, 混池对) / donor(血)。
    """
    import scanpy as sc

    hdir = PROJECT / "data" / "single-cell" / "human"
    bpath = hdir / "human_brain.h5ad"
    ppath = hdir / "human_blood.h5ad"

    log(f"加载人脑: {bpath.name}")
    a = sc.read_h5ad(bpath)
    Xb = a.X
    if hasattr(Xb, 'toarray'): Xb = Xb.toarray()
    X_brain = np.asarray(Xb, dtype=np.float32)
    gene_names = list(a.var_names.astype(str))
    brain_key = a.obs['disease'].astype(str).values
    brain_ct = a.obs['celltype'].astype(str).values
    brain_donor = a.obs['donor_pair'].astype(str).values
    log(f"  脑: {X_brain.shape}, 疾病: {dict(zip(*np.unique(brain_key, return_counts=True)))}, "
        f"类型: {dict(zip(*np.unique(brain_ct, return_counts=True)))}")

    log(f"加载人血: {ppath.name}")
    b = sc.read_h5ad(ppath)
    Xp = b.X
    if hasattr(Xp, 'toarray'): Xp = Xp.toarray()
    X_blood = np.asarray(Xp, dtype=np.float32)
    assert list(b.var_names.astype(str)) == gene_names, "人血基因顺序与人脑不一致!"
    blood_key = b.obs['disease'].astype(str).values
    blood_ct = b.obs['celltype'].astype(str).values
    blood_donor = b.obs['donor'].astype(str).values
    log(f"  血: {X_blood.shape}, 疾病: {dict(zip(*np.unique(blood_key, return_counts=True)))}, "
        f"类型: {dict(zip(*np.unique(blood_ct, return_counts=True)))}")
    log("配对键: 疾病标签 (AD/CTRL, 人侧无个体配对)")

    return (X_brain, X_blood, gene_names, brain_key, blood_key,
            brain_ct, blood_ct, brain_donor, blood_donor)


def _load_member(pair_key, species):
    """member-level 加载: data/member_level/*.h5ad (列=shared OG + 私有节点, 标签已烘焙)。"""
    import scanpy as sc
    mdir = PROJECT / "data" / "member_level"

    def read(name):
        a = sc.read_h5ad(mdir / name)
        X = a.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32), a

    if species == "human":
        X_brain, a = read("human_brain.h5ad")
        X_blood, b = read("human_blood.h5ad")
        gene_names = list(a.var_names.astype(str))
        assert list(b.var_names.astype(str)) == gene_names, "人血 member 列与人脑不一致!"
        # v5.2: 血端优先用 Leiden 簇级细分标签 (CD4T/CD8T 拆分)
        blood_ct_col = "celltype_leiden" if "celltype_leiden" in b.obs.columns else "celltype"
        log(f"[member] 人脑: {X_brain.shape}, 人血: {X_blood.shape}, 节点 {len(gene_names)}"
            f" (血标签列={blood_ct_col})")
        log("配对键: 疾病标签 (AD/CTRL, 人侧无个体配对)")
        return (X_brain, X_blood, gene_names,
                a.obs["disease"].astype(str).values, b.obs["disease"].astype(str).values,
                a.obs["celltype"].astype(str).values, b.obs[blood_ct_col].astype(str).values,
                a.obs["donor_pair"].astype(str).values, b.obs["donor"].astype(str).values)

    X_brain, a = read("mouse_brain.h5ad")
    X1, b1 = read("mouse_blood_bm12.h5ad")
    X2, b2 = read("mouse_blood_bm34.h5ad")
    gene_names = list(a.var_names.astype(str))
    assert list(b1.var_names.astype(str)) == gene_names and \
        list(b2.var_names.astype(str)) == gene_names, "鼠血 member 列与鼠脑不一致!"
    X_blood = np.vstack([X1, X2])
    brain_geno = a.obs["genotype"].astype(str).values
    brain_mouse = a.obs["mouse"].astype(str).values
    blood_geno = np.concatenate([b1.obs["genotype"].astype(str).values,
                                 b2.obs["genotype"].astype(str).values])
    # BM12 的 mouse 形如 5xFAD_1_femur → 个体 5xFAD_1; BM34 已是 5xFAD_1 形式
    m1 = pd.Series(b1.obs["mouse"].astype(str).values).map(lambda s: s.rsplit("_", 1)[0]).values
    blood_mouse = np.concatenate([m1, b2.obs["mouse"].astype(str).values])
    brain_ct = a.obs["celltype"].astype(str).values
    blood_ct = np.concatenate([b1.obs["celltype"].astype(str).values,
                               b2.obs["celltype"].astype(str).values])
    log(f"[member] 鼠脑: {X_brain.shape}, 鼠血(BM12+BM34): {X_blood.shape}, 节点 {len(gene_names)}")
    if pair_key == "mouse":
        log("配对键: 鼠个体 (个体配对)")
        brain_key, blood_key = brain_mouse, blood_mouse
    else:
        log("配对键: 基因型")
        brain_key, blood_key = brain_geno, blood_geno
    return (X_brain, X_blood, gene_names, brain_key, blood_key, brain_ct, blood_ct,
            brain_mouse, blood_mouse)


def load_data_joint(pair_key="mouse", member_level=False):
    """v5 联合模式: 加载双物种, 返回结构化的双物种数据包。

    返回 dict:
      genes, 以及每个物种的 (X_brain, X_blood, key(配对), ct(对齐), ct_raw(原标签), donor)
    """
    Xm_b, Xm_p, genes, m_bkey, m_pkey, m_bct, m_pct, m_bdonor, m_pdonor = \
        load_data(pair_key=pair_key, species="mouse", member_level=member_level)
    Xh_b, Xh_p, genes_h, h_bkey, h_pkey, h_bct, h_pct, h_bdonor, h_pdonor = \
        load_data(species="human", member_level=member_level)
    assert genes == genes_h, "人鼠基因空间不一致!"
    return {
        "genes": genes,
        "mouse": {
            "X_brain": Xm_b, "X_blood": Xm_p,
            "key": np.asarray(m_bkey).astype(str), "key_blood": np.asarray(m_pkey).astype(str),
            "ct_raw": np.asarray(m_bct).astype(str), "ct_blood_raw": np.asarray(m_pct).astype(str),
            "ct": align_ct(m_bct, "mouse"), "ct_blood": align_ct(m_pct, "mouse"),
            "donor": m_bdonor, "donor_blood": m_pdonor,
        },
        "human": {
            "X_brain": Xh_b, "X_blood": Xh_p,
            "key": np.asarray(h_bkey).astype(str), "key_blood": np.asarray(h_pkey).astype(str),
            "ct_raw": np.asarray(h_bct).astype(str), "ct_blood_raw": np.asarray(h_pct).astype(str),
            "ct": align_ct(h_bct, "human"), "ct_blood": align_ct(h_pct, "human"),
            "donor": h_bdonor, "donor_blood": h_pdonor,
        },
    }


# ═══════════════════════════════════════════════════════════════
# 训练(v6: 仅 train_joint,旧版 forward-only train() 已删除)
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# v5 联合训练: 人鼠共享动力学核心, 物种+对齐细胞类型条件
# ═══════════════════════════════════════════════════════════════

def train_joint(data, esm_matrix, config, mouse_frac=0.5):
    """人鼠联合训练。data 为 load_data_joint() 的返回。

    cond = [species(2) | brain_align | blood_align]
    鼠: 同个体配对 + 小胶质区室限定; 人: 同疾病标签配对。
    (2026-08-09: 零分布(shuffle/null)机制已废除并移除, 预筛统一走 randko)
    """
    log("=" * 60)
    log("训练 v5 联合模型(人鼠共享动力学 + 对齐细胞类型条件)")
    log("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)   # bit 级可复现: 模型初始化固定
    np.random.seed(config.seed)      # bit 级可复现: 采样固定
    genes = data["genes"]

    COND_DIM = 2 + len(BRAIN_ALIGN) + len(BLOOD_ALIGN)
    config.cond_dim = COND_DIM
    config.ct_brain_labels = BRAIN_ALIGN   # 供 ko_edges.py 复用
    config.ct_blood_labels = BLOOD_ALIGN

    def oh_table(ct_aligned, label_list, n_rows):
        idx = {l: i for i, l in enumerate(label_list)}
        M = np.zeros((n_rows, len(label_list)), np.float32)
        ok = np.array([l in idx for l in ct_aligned])
        M[np.where(ok)[0], [idx[l] for l in ct_aligned[ok]]] = 1.0
        return M

    conds = {}
    for sp_i, sp in enumerate(["mouse", "human"]):
        d = data[sp]
        Bb = oh_table(d["ct"], BRAIN_ALIGN, len(d["ct"]))
        Bp = oh_table(d["ct_blood"], BLOOD_ALIGN, len(d["ct_blood"]))
        Cb = np.zeros((len(d["ct"]), COND_DIM), np.float32)
        Cb[:, sp_i] = 1.0                      # species 位只放在脑表
        Cb[:, 2:2 + len(BRAIN_ALIGN)] = Bb
        Cp = np.zeros((len(d["ct_blood"]), COND_DIM), np.float32)
        Cp[:, 2 + len(BRAIN_ALIGN):] = Bp      # 血表只含血型块
        conds[sp] = (torch.from_numpy(Cb).to(device), torch.from_numpy(Cp).to(device))

    model = WorldModelV3(len(genes), esm_matrix, config).to(device)
    # 双物种联合中心化
    gm = np.concatenate([data["mouse"]["X_brain"], data["mouse"]["X_blood"],
                         data["human"]["X_brain"], data["human"]["X_blood"]], 0) \
        .mean(axis=0).astype(np.float32)
    model.gene_mean.copy_(torch.from_numpy(gm).to(device))

    # 配对池
    m = data["mouse"]
    m_pool = {g: np.where(m["key_blood"] == g)[0] for g in np.unique(m["key_blood"])}
    my_mask = np.isin(m["ct_blood_raw"], config.myeloid_compartment)
    m_my_pool = {g: idx[my_mask[idx]] for g, idx in m_pool.items()}
    h = data["human"]
    h_pool = {g: np.where(h["key_blood"] == g)[0] for g in np.unique(h["key_blood"])}
    log(f"鼠池 {len(m_pool)} 键(髓系限定 {len(m_my_pool)}), 人池 {len(h_pool)} 键, cond_dim={COND_DIM}")

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.lr)
    rng = np.random.RandomState(config.seed)  # bit 级可复现: 采样固定
    t0 = time.time()
    log_rows = []
    best_loss, best_state = float("inf"), None
    Xm_b, Xm_p = m["X_brain"], m["X_blood"]
    Xh_b, Xh_p = h["X_brain"], h["X_blood"]
    Cm_b, Cm_p = conds["mouse"]
    Ch_b, Ch_p = conds["human"]

    # GPU 常驻数据 (2026-08-08 提速): 四矩阵 ~12GB, 48GB 显存放得下。
    # 训练语义不变(同 batch_size/steps/种子/配对抽样), 仅消除逐步 H2D 拷贝与 CPU numpy 索引。
    def _to_gpu(X):
        try:
            return torch.from_numpy(X).float().to(device)
        except RuntimeError:
            return None
    Xm_b_g = _to_gpu(Xm_b)
    Xm_p_g = _to_gpu(Xm_p)
    Xh_b_g = _to_gpu(Xh_b)
    Xh_p_g = _to_gpu(Xh_p)
    gpu_resident = all(t is not None for t in (Xm_b_g, Xm_p_g, Xh_b_g, Xh_p_g))
    log(f"数据常驻 GPU: {'是 (~12GB)' if gpu_resident else '否(显存不足, 回退逐步拷贝)'}")

    def _fetch(Xg, Xc, idx):
        if Xg is not None:
            return Xg.index_select(0, torch.from_numpy(idx).to(device))
        return torch.from_numpy(Xc[idx]).float().to(device)

    def batch(sp):
        if sp == "mouse":
            bi = rng.choice(len(Xm_b), config.batch_size, replace=False)
            bg = m["key"][bi]
            di = np.empty(config.batch_size, np.int64)
            is_micro = np.isin(m["ct_raw"][bi], config.micro_labels)
            for g in np.unique(bg):
                sel_g = bg == g
                sel_m = sel_g & is_micro
                if sel_m.any():
                    pool = m_my_pool[g]
                    di[sel_m] = pool[rng.randint(0, len(pool), int(sel_m.sum()))]
                sel_o = sel_g & ~is_micro
                if sel_o.any():
                    pool = m_pool[g]
                    di[sel_o] = pool[rng.randint(0, len(pool), int(sel_o.sum()))]
            x_b = _fetch(Xm_b_g, Xm_b, bi)
            x_p = _fetch(Xm_p_g, Xm_p, di)
            cond = Cm_b[bi] + Cm_p[di]
            return x_b, x_p, cond
        bi = rng.choice(len(Xh_b), config.batch_size, replace=False)
        bg = h["key"][bi]
        di = np.empty(config.batch_size, np.int64)
        for g in np.unique(bg):
            sel = bg == g
            pool = h_pool[g]
            di[sel] = pool[rng.randint(0, len(pool), int(sel.sum()))]
        x_b = _fetch(Xh_b_g, Xh_b, bi)
        x_p = _fetch(Xh_p_g, Xh_p, di)
        cond = Ch_b[bi] + Ch_p[di]
        return x_b, x_p, cond

    log("══ 交替人鼠 batch 训练开始 ══")
    n_mouse_steps = int(config.steps_per_epoch * mouse_frac)
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for step in range(config.steps_per_epoch):
            x_b, x_p, cond = batch("mouse" if step < n_mouse_steps else "human")
            loss_dict = model.compute_loss(x_b, x_p, cond)
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss_dict)

        avg = {k: float(np.mean([l[k].item() if torch.is_tensor(l[k]) else l[k] for l in losses]))
               for k in losses[0]}
        log_rows.append({"epoch": epoch, **avg})
        if avg["loss"] < best_loss:
            best_loss = avg["loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            log(f"  Epoch {epoch:3d}/{config.epochs} | loss={avg['loss']:.4f} "
                f"mse={avg['mse']:.4f} cos={avg['cos']:.4f} recon={avg['recon']:.4f} "
                f"drift={avg['drift']:.4f} | {time.time()-t0:.0f}s")

    if best_state:
        model.load_state_dict(best_state)

    out_dir = PROJECT / config.output_dir
    model_dir = PROJECT / config.model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "gene_names": genes,
        "esm_matrix": esm_matrix,
        "joint": True,
        "cond_layout": {"species": ["mouse", "human"],
                        "brain_align": BRAIN_ALIGN, "blood_align": BLOOD_ALIGN},
    }, model_dir / "model.pt")
    pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)
    log(f"模型保存: {model_dir / 'model.pt'}")
    return model



# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "train-test"], default="train-test")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-genes", type=int, default=None)
    parser.add_argument("--n-cells", type=int, default=1000)
    parser.add_argument("--pair-key", choices=["mouse", "genotype"], default="mouse",
                        help="配对键(仅鼠): mouse=个体配对(v3.4 默认), genotype=基因型配对(v3.3)")
    parser.add_argument("--species", choices=["mouse", "human"], default="mouse",
                        help="物种: human 时配对键固定为疾病标签 AD/CTRL")
    parser.add_argument("--joint", action="store_true", default=True,
                        help="v5 联合训练: 人鼠共享动力学核心, 物种+对齐细胞类型条件")
    parser.add_argument("--member-level", action="store_true", default=True,
                        help="member-level 列空间(data/member_level/, OG 求和 + 私有节点); "
                             "joint 模式下模型存 models/world_model_v5_joint_member")
    parser.add_argument("--beta-rev", type=float, default=1.0,
                        help="v5.1 反向(血→脑)预测损失权重; >0 时模型存 *_v5_1_* 目录")
    parser.add_argument("--separate-rev-drift", action="store_true", default=True,
                        help="v5.2: 独立反向漂移头(需配合 --beta-rev>0 使用); 模型存 *_v5_2_* 目录")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="覆盖 batch_size (配合 GPU 常驻数据, 1024+ 可充分利用显存)")
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="覆盖 steps_per_epoch")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    cfg = Config()
    if args.epochs: cfg.epochs = args.epochs
    if args.beta_rev > 0: cfg.beta_rev = args.beta_rev
    if args.separate_rev_drift: cfg.separate_rev_drift = True
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.steps_per_epoch: cfg.steps_per_epoch = args.steps_per_epoch

    log("=" * 60)
    log("跨物种世界模型 v3(半训练 Decoder, 专家方案 2-A)")
    log("=" * 60)

    # v6: 纯 reverse 联合训练(人鼠共享动力学 + 对齐细胞类型条件)
    # forward 路线已删除,不再支持 beta_rev=0 的 forward-only 模式
    cfg.output_dir = "results/step1_world_model_v5_2_member"
    cfg.model_dir = "models/world_model_v5_2_joint_member"
    data = load_data_joint(pair_key=args.pair_key, member_level=args.member_level)
    esm_matrix = build_esm2_matrix(data["genes"], member_level=args.member_level)
    train_joint(data, esm_matrix, cfg)
    log("=" * 60)
    log("v6 完成(联合训练结束, 边提取请走 step2: ko_edges.py)")
    log("=" * 60)


if __name__ == "__main__":
    raise SystemExit("Use full-scope train.py; copied legacy training entry point disabled.")
