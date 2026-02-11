import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
import os

# --- 数据生成 ---
def sample_batch(batch_size, noise_level=0.25):
    """
    从 Swiss Roll 数据集中生成一批样本。
    """
    data, _ = make_swiss_roll(n_samples=batch_size, noise=noise_level)
    data = data[:, [0, 2]]
    data = data / 10.0
    return data

# --- MLP 模型定义 ---
class MLP(nn.Module):
    def __init__(self, N=40, data_dim=2, hidden_dim=256):
        """
        初始化 MLP 模型。

        改动：
        - 输入维度改为 data_dim + 1（拼接归一化时间步）
        - 输出维度改为 data_dim（预测噪声，而非均值+方差）
        - 激活函数改为 Tanh（更适合回归任务）
        - 使用单一网络而非 N 个 tail（参数共享，泛化更好）
        """
        super(MLP, self).__init__()
        self.N = N
        self.network_head = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # 单一 tail，所有时间步共享（通过输入 t 区分）
        self.network_tail = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, data_dim)  # 输出预测噪声
        )

    def forward(self, x, t):
        """
        MLP 模型的前向传播。

        参数:
            x (torch.Tensor): 输入数据张量 [Batch, data_dim]
            t (torch.Tensor or int): 时间步，支持标量或 [Batch] 张量
        """
        # 将 t 转为归一化的浮点张量 [Batch, 1]
        if isinstance(t, int):
            t_tensor = torch.full((x.shape[0], 1), t / self.N, device=x.device, dtype=x.dtype)
        else:
            t_tensor = t.view(-1, 1).float() / self.N
        # 拼接 x 和 t: [Batch, data_dim + 1]
        h = self.network_head(torch.cat([x, t_tensor], dim=1))
        return self.network_tail(h)

# --- 扩散模型定义 ---
class DiffusionModel(nn.Module):
    def __init__(self, model: nn.Module, n_steps=40, device='cpu'):
        """
        初始化扩散模型。
        """
        super().__init__()
        self.model = model
        self.device = device

        # 线性 Beta 调度
        betas = torch.linspace(1e-4, 0.02, n_steps)
        self.beta = betas.to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps

        # 预计算常用系数
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bar)

    def forward_process(self, x0, t):
        """
        执行前向扩散过程（加噪）。

        改动：t 现在是 [Batch] 的张量，支持每个样本不同时间步。
        返回：(xt, noise, t) 替代原来的 (mu_posterior, sigma_posterior, xt)
        """
        noise = torch.randn_like(x0)
        sqrt_a = self.sqrt_alpha_bar[t].view(-1, 1)
        sqrt_one_minus_a = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)
        xt = sqrt_a * x0 + sqrt_one_minus_a * noise
        return xt, noise, t

    def reverse(self, xt, t):
        """
        执行反向扩散过程（去噪）- 用于采样。

        参数:
            xt: [Batch, 2] 当前带噪数据
            t (int): 当前时间步（从 n_steps-1 到 0）
        """
        t_tensor = torch.full((xt.shape[0],), t, dtype=torch.long, device=self.device)
        eps_theta = self.model(xt, t_tensor)

        alpha = self.alpha[t]
        alpha_bar = self.alpha_bar[t]
        beta = self.beta[t]

        # 标准 DDPM 采样公式
        mean = (1 / torch.sqrt(alpha)) * (
            xt - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps_theta
        )

        if t == 0:
            return mean, None, mean
        else:
            sigma = torch.sqrt(beta)
            z = torch.randn_like(xt)
            return mean, sigma, mean + sigma * z

    @torch.no_grad()
    def sample(self, size):
        """
        从模型中生成样本。
        """
        noise = torch.randn((size, 2), device=self.device)
        samples = [noise]
        for t in range(self.n_steps - 1, -1, -1):
            _, _, x = self.reverse(samples[-1], t)
            samples.append(x)
        return samples

# --- 训练函数 ---
def train(model, optimizer, nb_epochs=10000, batch_size=2048,
          noise_level=0.1, save_freq=2000, output_dir='Imgs', device='cpu'):
    """
    训练扩散模型。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mse_loss = nn.MSELoss()
    model.train()
    progress_bar = tqdm(range(nb_epochs))
    for step in progress_bar:
        x0 = torch.from_numpy(sample_batch(batch_size, noise_level)).float().to(device)

        # 每个样本随机采样一个时间步
        t = torch.randint(0, model.n_steps, (batch_size,)).to(device)

        # 前向加噪
        xt, noise, t = model.forward_process(x0, t)

        # 模型预测噪声
        pred_noise = model.model(xt, t)

        # MSE 损失
        loss = mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 100 == 0:
            progress_bar.set_description(f"Loss: {loss.item():.4f}")

        # 定期保存图片
        if step % save_freq == 0:
            plot(model, save_path=f"{output_dir}/step_{step}.png", show=False)
            model.train()

# --- 绘图函数 ---
@torch.no_grad()
def plot(model, save_path="Imgs/diffusion_model.png", show=True):
    """
    绘制扩散过程和生成的样本。
    """
    model.eval()
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(10, 6))
    x0 = sample_batch(5000)
    x0_tensor = torch.from_numpy(x0).float().to(model.device)

    # 前向加噪展示
    t_mid = torch.full((5000,), model.n_steps // 2, dtype=torch.long, device=model.device)
    t_end = torch.full((5000,), model.n_steps - 1, dtype=torch.long, device=model.device)
    x_mid, _, _ = model.forward_process(x0_tensor, t_mid)
    x_end, _, _ = model.forward_process(x0_tensor, t_end)
    data = [x0, x_mid.cpu().numpy(), x_end.cpu().numpy()]

    for i, title in enumerate([r'$t=0$', r'$t=\frac{T}{2}$', r'$t=T$']):
        plt.subplot(2, 3, 1 + i)
        plt.scatter(data[i][:, 0], data[i][:, 1], alpha=.1, s=1)
        plt.xlim([-2, 2]); plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if i == 0: plt.ylabel(r'$q(\mathbf{x})$', fontsize=17, rotation=0, labelpad=40)
        plt.title(title, fontsize=17)

    # 反向生成展示
    samples = model.sample(5000)
    n_samples = len(samples)
    for i, t_idx in enumerate([0, n_samples // 2, n_samples - 1]):
        plt.subplot(2, 3, 4 + i)
        plt.scatter(samples[t_idx][:, 0].cpu().numpy(),
                    samples[t_idx][:, 1].cpu().numpy(), alpha=.1, s=1, c='r')
        plt.xlim([-2, 2]); plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if i == 0: plt.ylabel(r'$p(\mathbf{x})$', fontsize=17, rotation=0, labelpad=40)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"结果已保存至 {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

# --- 主程序入口 ---
if __name__ == "__main__":
    # ===== 【所有可设置参数】 =====
    N_STEPS = 80              # 扩散总步数（增加步数，采样更精细）
    HIDDEN_DIM = 256          # 隐藏层维度
    DATA_DIM = 2              # 数据维度
    NB_EPOCHS = 10001         # 训练轮数
    BATCH_SIZE = 1024         # 批大小
    LR = 1e-3                 # 学习率
    NOISE_LEVEL = 0.1         # 数据噪声
    SAVE_FREQ = 1000          # 图片保存频率
    OUTPUT_DIR = 'Diffussion/swiss_roll_outputs'       # 图片保存目录
    # =============================

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化 MLP 和扩散模型
    model_mlp = MLP(N=N_STEPS, data_dim=DATA_DIM, hidden_dim=HIDDEN_DIM).to(device)
    diffusion_model = DiffusionModel(model_mlp, n_steps=N_STEPS, device=device)

    # 设置优化器
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=LR)

    # 开始训练
    print(f"正在 {device} 上开始训练...")
    train(diffusion_model, optimizer, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
          noise_level=NOISE_LEVEL, save_freq=SAVE_FREQ, output_dir=OUTPUT_DIR, device=device)

    # 绘图并保存最终结果
    plot(diffusion_model)