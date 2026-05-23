"""Vanilla GAN for tabular data (Credit Card Fraud)."""
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


class Discriminator(nn.Module):
    def __init__(self, data_dim=29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_vanilla_gan(loader, device, z_dim=100, data_dim=29, epochs=300, lr=2e-4,
                      save_path="outputs/vanilla_gan.pt",
                      plot_path="outputs/plots/vanilla_gan_loss.png"):
    G = Generator(z_dim, data_dim).to(device)
    D = Discriminator(data_dim).to(device)
    bce = nn.BCELoss()
    g_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    g_losses, d_losses = [], []

    for epoch in range(epochs):
        g_epoch, d_epoch = [], []
        for real, _ in loader:
            real = real.to(device)
            bs = real.size(0)
            real_lbl = torch.ones(bs, 1, device=device)
            fake_lbl = torch.zeros(bs, 1, device=device)

            # Train D
            z = torch.randn(bs, z_dim, device=device)
            fake = G(z).detach()
            d_loss = bce(D(real), real_lbl) + bce(D(fake), fake_lbl)
            d_opt.zero_grad(); d_loss.backward(); d_opt.step()

            # Train G
            z = torch.randn(bs, z_dim, device=device)
            g_loss = bce(D(G(z)), real_lbl)
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()

            g_epoch.append(g_loss.item())
            d_epoch.append(d_loss.item())

        g_losses.append(sum(g_epoch) / len(g_epoch))
        d_losses.append(sum(d_epoch) / len(d_epoch))

        if (epoch + 1) % 50 == 0:
            print(f"[Vanilla GAN] Epoch {epoch+1}/{epochs}  "
                  f"D: {d_losses[-1]:.4f}  G: {g_losses[-1]:.4f}")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(G.state_dict(), save_path)
    print(f"[Vanilla GAN] Model saved to {save_path}")

    # Plot
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")
    ax.plot(g_losses, label="Generator", color="#A8C8E8", linewidth=2)
    ax.plot(d_losses, label="Discriminator", color="#F4A8A8", linewidth=2)
    ax.set_xlabel("Epoch", color="#555555")
    ax.set_ylabel("Loss", color="#555555")
    ax.set_title("Vanilla GAN Training Loss", color="#333333")
    ax.legend()
    ax.grid(True, color="#FFFFFF", linewidth=1.0)
    ax.tick_params(colors="#666666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    print(f"[Vanilla GAN] Loss curve saved to {plot_path}")

    return G, g_losses, d_losses


@torch.no_grad()
def generate_samples(G, n, z_dim, device):
    G.eval()
    z = torch.randn(n, z_dim, device=device)
    return G(z).cpu()
