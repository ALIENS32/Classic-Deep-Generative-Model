import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
import os

# ============================================================
#  数据生成
#  从 sklearn 提供的三维瑞士卷中提取二维螺旋平面，
#  作为扩散模型的训练数据 q(x_0)。
# ============================================================

def sample_batch(batch_size, noise_level=0.25):
    """
    从瑞士卷（Swiss Roll）分布中采样一批二维数据点。

    流程：
        1. 调用 sklearn 生成三维瑞士卷点云 [batch_size, 3]
        2. 取第 0、2 列（x 轴和 z 轴），丢弃 y 轴，投影为二维螺旋
        3. 除以 10 进行归一化，使数据范围大致落在 [-1.5, 1.5]

    参数:
        batch_size  (int)  : 采样点数
        noise_level (float): 瑞士卷噪声幅度，控制数据点偏离理想螺旋带的程度

    返回:
        np.ndarray, 形状 [batch_size, 2]，一批来自 q(x_0) 的真实样本
    """
    data, _ = make_swiss_roll(n_samples=batch_size, noise=noise_level)
    data = data[:, [0, 2]]      # 三维 → 二维螺旋
    data = data / 10.0           # 归一化到网络友好的数值范围
    return data

# ============================================================
#  噪声预测网络（MLP）
#  接收加噪数据 x_t 与归一化时间步 t/T，输出预测噪声 ε̂_θ(x_t, t)。
#  所有时间步共享同一组参数，依靠输入中的 t/T 区分扩散阶段。
# ============================================================

class MLP(nn.Module):
    def __init__(self, N=40, data_dim=2, hidden_dim=256):
        """
        初始化噪声预测网络。

        网络结构：
            network_head : Linear(data_dim+1, hidden_dim) → Tanh
                           → Linear(hidden_dim, hidden_dim) → Tanh
            network_tail : Linear(hidden_dim, hidden_dim) → Tanh
                           → Linear(hidden_dim, data_dim)

        设计要点：
            - 输入维度 data_dim + 1：2 维数据拼接 1 维归一化时间步
            - 输出维度 data_dim：直接预测噪声 ε，而非均值+方差
            - 激活函数 Tanh：输出无界但平滑，适合回归任务
            - 单一 tail（参数共享）替代 N 个独立 tail，泛化能力更强

        参数:
            N          (int): 扩散总步数 T，用于时间步归一化 t/T ∈ [0, 1]
            data_dim   (int): 数据维度（瑞士卷为 2）
            hidden_dim (int): 隐藏层宽度
        """
        super(MLP, self).__init__()
        self.N = N

        # ---------- 编码器：将 (x_t, t/T) 映射到隐表示 ----------
        self.network_head = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # ---------- 解码器：从隐表示输出预测噪声 ε̂_θ ----------
        self.network_tail = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, data_dim)   # 输出与 x_0 同维的噪声预测
        )

    def forward(self, x, t):
        """
        前向传播：给定加噪数据和时间步，预测其中的噪声。

        参数:
            x (Tensor): [B, data_dim]，加噪后的数据 x_t
            t (int 或 Tensor):
                - int：采样阶段，所有样本共享同一时间步
                - Tensor [B]：训练阶段，每个样本各自的时间步

        返回:
            Tensor [B, data_dim]，预测噪声 ε̂_θ(x_t, t)
        """
        # ---- 时间步处理：统一转为归一化浮点张量 [B, 1] ----
        if isinstance(t, int):
            # 标量 → 全部填充同一值（采样阶段）
            t_tensor = torch.full(
                (x.shape[0], 1), t / self.N,
                device=x.device, dtype=x.dtype
            )
        else:
            # [B] 张量 → reshape 并归一化（训练阶段）
            t_tensor = t.view(-1, 1).float() / self.N

        # ---- 拼接数据与时间 [B, data_dim+1]，依次通过 head 和 tail ----
        h = self.network_head(torch.cat([x, t_tensor], dim=1))
        return self.network_tail(h)

# ============================================================
#  扩散模型（DiffusionModel）
#  管理前向加噪 / 反向去噪 / 噪声调度 / 完整采样。
# ============================================================

class DiffusionModel(nn.Module):
    def __init__(self, model: nn.Module, n_steps=40,
                 begin_beta=1e-4, end_beta=0.02, device='cpu'):
        """
        初始化扩散模型，预计算所有与噪声调度相关的系数。

        参数:
            model      (nn.Module): 噪声预测网络 ε_θ（MLP 实例）
            n_steps    (int)      : 扩散总步数 T
            begin_beta (float)    : β 调度起始值 β_1
            end_beta   (float)    : β 调度终止值 β_T
            device     (str)      : 运行设备 ('cpu' / 'cuda')
        """
        super().__init__()
        self.model = model
        self.device = device

        # ---- 线性 β 调度：β_1, β_2, ..., β_T ----
        betas = torch.linspace(begin_beta, end_beta, n_steps)
        self.beta = betas.to(device)                             # [T]
        self.alpha = 1. - self.beta                              # α_t = 1 - β_t
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)        # ᾱ_t = ∏_{s=1}^{t} α_s
        self.n_steps = n_steps

        # ---- 预计算前向加噪公式中的两个系数，避免重复开方 ----
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)              # √ᾱ_t  （信号系数）
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bar)  # √(1-ᾱ_t)（噪声系数）

    # ----------------------------------------------------------
    #  前向过程（加噪）
    #  利用重参数化技巧，从 x_0 一步跳到任意时刻 x_t：
    #      x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,  ε ~ N(0, I)
    # ----------------------------------------------------------
    def forward_process(self, x0, t):
        """
        对原始数据执行前向加噪。

        参数:
            x0 (Tensor): [B, 2]，干净数据 x_0
            t  (Tensor): [B]，每个样本各自的目标时间步（长整型）

        返回:
            xt    (Tensor): [B, 2]，加噪后的 x_t
            noise (Tensor): [B, 2]，本次采样的真实噪声 ε（训练标签）
            t     (Tensor): [B]，原样返回，便于后续传递
        """
        noise = torch.randn_like(x0)                                    # ε ~ N(0, I)
        sqrt_a = self.sqrt_alpha_bar[t].view(-1, 1)                      # √ᾱ_t → [B, 1]
        sqrt_one_minus_a = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)  # √(1-ᾱ_t) → [B, 1]
        xt = sqrt_a * x0 + sqrt_one_minus_a * noise                     # 重参数化加噪
        return xt, noise, t

    # ----------------------------------------------------------
    #  反向过程（单步去噪）
    #  DDPM 采样公式：
    #      x_{t-1} = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t) · ε̂_θ) + σ_t · z
    #  其中 σ_t = √β_t，z ~ N(0, I)；t=0 时不加噪声。
    # ----------------------------------------------------------
    def reverse(self, xt, t):
        """
        执行一步反向去噪采样。

        参数:
            xt (Tensor): [B, 2]，当前时刻的带噪数据 x_t
            t  (int)   : 当前时间步（从 T-1 递减到 0）

        返回:
            mean   (Tensor)     : [B, 2]，去噪均值 μ_θ(x_t, t)
            sigma  (Tensor/None): 标量噪声标准差 σ_t，t=0 时为 None
            x_prev (Tensor)     : [B, 2]，采样得到的 x_{t-1}
        """
        # 将标量 t 扩展为 [B] 张量，供 MLP 使用
        t_tensor = torch.full(
            (xt.shape[0],), t, dtype=torch.long, device=self.device
        )

        # 噪声预测网络推理
        eps_theta = self.model(xt, t_tensor)        # ε̂_θ(x_t, t), [B, 2]

        # 取当前时间步的调度系数（标量）
        alpha     = self.alpha[t]
        alpha_bar = self.alpha_bar[t]
        beta      = self.beta[t]

        # ---- 计算去噪均值 μ_θ ----
        #   μ = (1/√α_t) · (x_t - (1-α_t)/√(1-ᾱ_t) · ε̂_θ)
        mean = (1 / torch.sqrt(alpha)) * (
            xt - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps_theta
        )

        if t == 0:
            # 最后一步：直接返回均值作为生成结果 x_0，不再注入噪声
            return mean, None, mean
        else:
            # 中间步：加入随机噪声以保持反向马尔可夫链的随机性
            sigma = torch.sqrt(beta)                # σ_t = √β_t
            z = torch.randn_like(xt)                # z ~ N(0, I)
            return mean, sigma, mean + sigma * z    # x_{t-1} = μ + σ_t · z

    # ----------------------------------------------------------
    #  完整采样（Algorithm 2）
    #  从纯噪声 x_T ~ N(0, I) 出发，逐步去噪直至 x_0。
    # ----------------------------------------------------------
    @torch.no_grad()
    def sample(self, size):
        """
        从模型生成一批样本。

        参数:
            size (int): 需要生成的样本数量

        返回:
            samples (list[Tensor]):
                长度 T+1 的列表。
                samples[0]  = x_T（纯噪声）
                samples[-1] = x_0（最终生成结果）
                中间元素为反向过程各时刻的快照，可用于可视化。
        """
        noise = torch.randn((size, 2), device=self.device)   # x_T ~ N(0, I)
        samples = [noise]
        for t in range(self.n_steps - 1, -1, -1):            # t: T-1 → 0
            _, _, x = self.reverse(samples[-1], t)
            samples.append(x)
        return samples

# ============================================================
#  训练循环
#  对应 DDPM Algorithm 1（Training）：
#      1. x_0 ~ q(x_0)
#      2. t  ~ Uniform{0, ..., T-1}
#      3. ε  ~ N(0, I)
#      4. 梯度下降 ‖ε - ε_θ(√ᾱ_t·x_0 + √(1-ᾱ_t)·ε, t)‖²
#      5. 重复直至收敛
# ============================================================

def train(model, optimizer, nb_epochs=10000, batch_size=2048,
          noise_level=0.1, save_freq=2000, output_dir='Imgs', device='cpu'):
    """
    训练扩散模型。

    参数:
        model       (DiffusionModel): 扩散模型实例
        optimizer   (Optimizer)     : 优化器（如 Adam）
        nb_epochs   (int)           : 训练总轮数
        batch_size  (int)           : 每轮采样的数据点数
        noise_level (float)         : 瑞士卷数据生成时的噪声幅度
        save_freq   (int)           : 每隔多少步保存一次可视化图片
        output_dir  (str)           : 图片保存目录
        device      (str)           : 运行设备

    副作用:
        - 就地更新 model 的参数
        - 定期在 output_dir 下保存生成效果图
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mse_loss = nn.MSELoss()
    model.train()
    progress_bar = tqdm(range(nb_epochs))

    for step in progress_bar:
        # ---- Step 1: 从瑞士卷分布采样真实数据 x_0 ----
        x0 = torch.from_numpy(
            sample_batch(batch_size, noise_level)
        ).float().to(device)                                      # [B, 2]

        # ---- Step 2: 为每个样本独立采样随机时间步 ----
        t = torch.randint(0, model.n_steps, (batch_size,)).to(device)  # [B]

        # ---- Step 3: 前向加噪，同时得到真实噪声作为标签 ----
        xt, noise, t = model.forward_process(x0, t)              # xt [B,2], noise [B,2]

        # ---- Step 4a: 噪声预测网络推理 ----
        pred_noise = model.model(xt, t)                           # ε̂_θ(x_t, t), [B, 2]

        # ---- Step 4b: 计算简化 MSE 损失 ‖ε - ε̂_θ‖² ----
        loss = mse_loss(pred_noise, noise)

        # ---- Step 4c: 反向传播与参数更新 ----
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止爆炸
        optimizer.step()

        # ---- 日志：每 100 步刷新进度条上的损失值 ----
        if step % 100 == 0:
            progress_bar.set_description(f"Loss: {loss.item():.4f}")

        # ---- 定期可视化：保存当前生成效果图 ----
        if step % save_freq == 0:
            plot(model, save_path=f"{output_dir}/step_{step}.png", show=False)
            model.train()   # plot() 内部会切到 eval 模式，这里切回 train

# ============================================================
#  绘图函数
#  上半部分：前向加噪过程 q(x) 在 t=0, T/2, T 三个时刻的点云
#  下半部分：反向生成过程 p(x) 在起点、中点、终点的点云
# ============================================================

@torch.no_grad()
def plot(model, save_path="Diffusion/Diffusion_Swiss_Roll/swiss_roll_outputs",
         show=True):
    """
    绘制扩散过程的 2×3 可视化面板并保存。

    布局:
        第一行（蓝色）：前向加噪 q(x) — t=0 | t=T/2 | t=T
        第二行（红色）：反向生成 p(x) — 纯噪声 | 中间态 | 最终生成

    参数:
        model     (DiffusionModel): 训练中的扩散模型
        save_path (str)           : 图片保存路径
        show      (bool)          : 是否弹窗显示（训练时设为 False）
    """
    model.eval()

    # 确保保存目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(10, 6))

    # ---- 准备真实数据 ----
    x0 = sample_batch(5000)
    x0_tensor = torch.from_numpy(x0).float().to(model.device)

    # ---- 前向加噪：分别加噪到 t=T/2 和 t=T ----
    t_mid = torch.full((5000,), model.n_steps // 2,
                       dtype=torch.long, device=model.device)
    t_end = torch.full((5000,), model.n_steps - 1,
                       dtype=torch.long, device=model.device)
    x_mid, _, _ = model.forward_process(x0_tensor, t_mid)
    x_end, _, _ = model.forward_process(x0_tensor, t_end)

    # ---- 第一行：绘制前向加噪的三个阶段（蓝色点云） ----
    data = [x0, x_mid.cpu().numpy(), x_end.cpu().numpy()]
    for i, title in enumerate([r'$t=0$', r'$t=\frac{T}{2}$', r'$t=T$']):
        plt.subplot(2, 3, 1 + i)
        plt.scatter(data[i][:, 0], data[i][:, 1], alpha=.1, s=1)
        plt.xlim([-2, 2]); plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if i == 0:
            plt.ylabel(r'$q(\mathbf{x})$', fontsize=17,
                        rotation=0, labelpad=40)
        plt.title(title, fontsize=17)

    # ---- 第二行：绘制反向生成的三个阶段（红色点云） ----
    samples = model.sample(5000)
    n_samples = len(samples)     # T+1 个快照
    for i, t_idx in enumerate([0, n_samples // 2, n_samples - 1]):
        plt.subplot(2, 3, 4 + i)
        plt.scatter(samples[t_idx][:, 0].cpu().numpy(),
                    samples[t_idx][:, 1].cpu().numpy(),
                    alpha=.1, s=1, c='r')
        plt.xlim([-2, 2]); plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if i == 0:
            plt.ylabel(r'$p(\mathbf{x})$', fontsize=17,
                        rotation=0, labelpad=40)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"结果已保存至 {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

# ============================================================
#  主程序入口
#  集中管理所有超参数，初始化模型与优化器，启动训练。
# ============================================================

if __name__ == "__main__":

    # ===== 【超参数配置区】 =====
    N_STEPS    = 80       # 扩散总步数 T（步数越多，采样越精细，但推理越慢）
    HIDDEN_DIM = 256      # MLP 隐藏层宽度
    DATA_DIM   = 2        # 数据维度（二维瑞士卷）
    NB_EPOCHS  = 10001    # 训练总轮数
    BATCH_SIZE = 1024     # 每轮训练的样本数
    LR         = 1e-3     # Adam 学习率
    NOISE_LEVEL = 0.1     # 瑞士卷数据生成噪声（越小螺旋越紧致）
    SAVE_FREQ  = 1000     # 每隔多少步保存一张可视化图
    OUTPUT_DIR = 'Diffusion/Diffusion_Swiss_Roll/swiss_roll_outputs'  # 图片输出目录
    BEGIN_BETA = 1e-4     # β 调度起始值（较小 → 前期加噪轻微）
    END_BETA   = 0.02     # β 调度终止值（较大 → 后期加噪显著）
    # ============================

    # 自动选择 GPU / CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- 构建噪声预测网络 ----
    model_mlp = MLP(
        N=N_STEPS, data_dim=DATA_DIM, hidden_dim=HIDDEN_DIM
    ).to(device)

    # ---- 构建扩散模型（包含调度系数与采样逻辑） ----
    diffusion_model = DiffusionModel(
        model_mlp, n_steps=N_STEPS,
        begin_beta=BEGIN_BETA, end_beta=END_BETA, device=device
    )

    # ---- 优化器 ----
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=LR)

    # ---- 启动训练 ----
    print(f"正在 {device} 上开始训练...")
    train(
        diffusion_model, optimizer,
        nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
        noise_level=NOISE_LEVEL, save_freq=SAVE_FREQ,
        output_dir=OUTPUT_DIR, device=device
    )