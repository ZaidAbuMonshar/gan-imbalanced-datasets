"""Wasserstein GAN (WGAN) for tabular data (Credit Card Fraud)."""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os


class Generator(nn.Module):
    def __init__(self, z_dim=100, data_dim=29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, data_dim),
        )

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    """No Sigmoid — outputs unbounded real-valued scores."""
    def __init__(self, data_dim=29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_wgan(loader, device, z_dim=100, data_dim=29, epochs=300, lr=5e-5,
               n_critic=5, clip_val=0.01,
               save_path="outputs/wgan.pt",
               plot_path="outputs/plots/wgan_loss.png"):
    G = Generator(z_dim, data_dim).to(device)
    C = Critic(data_dim).to(device)
    g_opt = torch.optim.RMSprop(G.parameters(), lr=lr)
    c_opt = torch.optim.RMSprop(C.parameters(), lr=lr)

    g_losses, c_losses = [], []

    for epoch in range(epochs):
        g_epoch, c_epoch = [], []
        for real, _ in loader:
            real = real.to(device)
            bs = real.size(0)

            # Train Critic n_critic times
            for _ in range(n_critic):
                z = torch.randn(bs, z_dim, device=device)
                fake = G(z).detach()
                c_loss = -(C(real).mean() - C(fake).mean())
                c_opt.zero_grad(); c_loss.backward(); c_opt.step()
                for p in C.parameters():
                    p.data.clamp_(-clip_val, clip_val)

            # Train Generator
            z = torch.randn(bs, z_dim, device=device)
            g_loss = -C(G(z)).mean()
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()

            g_epoch.append(g_loss.item())
            c_epoch.append(-c_loss.item())

        g_losses.append(sum(g_epoch) / len(g_epoch))
        c_losses.append(sum(c_epoch) / len(c_epoch))

        if (epoch + 1) % 50 == 0:
            print(f"[WGAN] Epoch {epoch+1}/{epochs}  "
                  f"W-dist: {c_losses[-1]:.4f}  G: {g_losses[-1]:.4f}")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(G.state_dict(), save_path)
    print(f"[WGAN] Model saved to {save_path}")

    # Plot
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")
    ax.plot(g_losses, label="Generator", color="#A8C8E8", linewidth=2)
    ax.plot(c_losses, label="Wasserstein distance", color="#F4A8A8", linewidth=2)
    ax.set_xlabel("Epoch", color="#555555")
    ax.set_ylabel("Loss", color="#555555")
    ax.set_title("WGAN Training Loss", color="#333333")
    ax.legend()
    ax.grid(True, color="#FFFFFF", linewidth=1.0)
    ax.tick_params(colors="#666666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    print(f"[WGAN] Loss curve saved to {plot_path}")

    return G, g_losses, c_losses


@torch.no_grad()
def generate_samples(G, n, z_dim, device):
    G.eval()
    z = torch.randn(n, z_dim, device=device)
    return G(z).cpu()
