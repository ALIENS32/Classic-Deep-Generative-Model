import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from tqdm import tqdm
import os

# ==========================================
# 1. 神经网络模型定义 (噪声预测器)
# ==========================================
class DiffusionMLP(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=256, n_steps=80):
        super().__init__()
        self.n_steps = n_steps
        # 输入维度: [Batch, 3] (x, y + t)
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim), 
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, data_dim) # 输出维度: [Batch, 2]
        )

    def forward(self, x, t):
        # t 归一化后维度: [Batch, 1]
        t_float = t.view(-1, 1).float() / self.n_steps 
        # 拼接后维度: [Batch, 3]
        combined = torch.cat([x, t_float], dim=1) 
        return self.net(combined)

# ==========================================
# 2. 扩散管理器 (物理过程模拟)
# ==========================================
class DiffusionManager:
    """
    作用：预计算扩散公式中的所有系数
    接口参数说明：
    - beta_start/beta_end: 控制噪声的起始和终点强度
    """
    def __init__(self, n_steps, beta_start, beta_end, device):
        self.n_steps = n_steps
        self.device = device
        
        # --- torch.linspace 接口 ---
        # 作用：生成 80 个从 beta_start 线性增长到 beta_end 的数值
        # 输出维度：[80]
        self.betas = torch.linspace(beta_start, beta_end, n_steps).to(device)
        
        # --- 计算 alpha ---
        self.alphas = 1.0 - self.betas
        
        # --- torch.cumprod 接口 ---
        # 作用：计算累乘 alpha_bar = alpha_0 * alpha_1 * ... * alpha_t
        # 输出维度：[80]
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # --- 预计算公式项 ---
        # 输出维度均为 [80]
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x0, t):
        """
        前向加噪接口
        x0 维度: [Batch, 2]
        t 维度: [Batch]
        ---
        输出 xt 维度: [Batch, 2]
        """
        noise = torch.randn_like(x0) 
        # 索引系数并调整维度为 [Batch, 1] 以便进行广播相乘
        sqrt_a = self.sqrt_alphas_cumprod[t].view(-1, 1) 
        sqrt_one_minus_a = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        # 公式: xt = sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*noise
        xt = sqrt_a * x0 + sqrt_one_minus_a * noise
        return xt, noise

# ==========================================
# 3. 训练与生成主流程
# ==========================================
def main():
    # --- 【可设置参数区】 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_steps = 80            # 扩散总步数
    beta_start = 1e-4       # 初始噪声强度 (建议 1e-4)
    beta_end = 0.02         # 最终噪声强度 (建议 0.02)
    batch_size = 1024       # 每批训练样本数
    epochs = 10001           # 训练总迭代次数
    save_freq = 1000         # 图片保存频率
    lr = 1e-3               # 学习率
    
    # 路径设置
    output_dir = 'DDPM-2015/swiss_roll_outputs'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # --- 初始化接口 ---
    model = DiffusionMLP(n_steps=n_steps).to(device)
    # 将 beta 参数传入管理器
    manager = DiffusionManager(n_steps, beta_start, beta_end, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_criterion = nn.MSELoss()

    print(f"任务启动 | 步数: {n_steps} | Beta: {beta_start}~{beta_end} | 设备: {device}")

    # --- 训练循环 ---
    pbar = tqdm(range(epochs), desc="Training")
    for step in pbar:
        # 1. 生成瑞士卷数据
        # raw_data 维度: [1024, 3] -> 切片后 x0 维度: [1024, 2]
        raw_data, _ = make_swiss_roll(n_samples=batch_size, noise=0.1)
        x0 = torch.from_numpy(raw_data[:, [0, 2]] / 10.0).float().to(device)
        
        # 2. 随机采样时间步 t，维度: [1024]
        t = torch.randint(0, n_steps, (batch_size,)).to(device)
        
        # 3. 前向加噪得到 xt [1024, 2]
        xt, noise = manager.add_noise(x0, t)
        
        # 4. 模型预测噪声，输入 [1024, 2] 和 [1024]，输出 [1024, 2]
        pred_noise = model(xt, t)
        
        # 5. 计算损失并反向传播
        loss = mse_criterion(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # 6. 定期保存图片
        if step % save_freq == 0:
            generate_and_save(model, manager, step, device, output_dir)

# ==========================================
# 4. 采样生成接口
# ==========================================
@torch.no_grad()
def generate_and_save(model, manager, step, device, output_dir):
    model.eval()
    # 起点是 2000 个纯噪声点，维度: [2000, 2]
    xt = torch.randn(2000, 2).to(device)
    
    # 反向迭代 T-1 -> 0
    for i in range(manager.n_steps - 1, -1, -1):
        # t 维度: [2000]
        t = torch.full((2000,), i, dtype=torch.long).to(device)
        eps_theta = model(xt, t) 
        
        alpha = manager.alphas[i]
        alpha_bar = manager.alphas_cumprod[i]
        beta = manager.betas[i]
        
        # 采样公式中的随机项
        z = torch.randn_like(xt) if i > 0 else 0
        
        # 反向迭代核心公式
        xt = (1 / torch.sqrt(alpha)) * (
            xt - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps_theta
        ) + torch.sqrt(beta) * z

    # 绘图并保存
    plt.figure(figsize=(6, 6))
    plt.scatter(xt[:, 0].cpu(), xt[:, 1].cpu(), s=5, alpha=0.6, c='teal')
    plt.title(f"Step {step} | Beta_end {manager.betas[-1].item():.2f}")
    plt.xlim(-2, 2); plt.ylim(-2, 2)
    plt.savefig(f'{output_dir}/step_{step}.png')
    plt.close()
    model.train()

if __name__ == "__main__":
    main()