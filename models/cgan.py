"""Conditional GAN (CGAN) for tabular data (Credit Card Fraud)."""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os


class Generator(nn.Module):
    def __init__(self, z_dim=100, data_dim=29, num_classes=2, embed_dim=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim + embed_dim, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, data_dim),
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        return self.net(torch.cat([z, c], dim=1))


class Discriminator(nn.Module):
    def __init__(self, data_dim=29, num_classes=2, embed_dim=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + embed_dim, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, x, labels):
        c = self.label_emb(labels)
        return self.net(torch.cat([x, c], dim=1))


def train_cgan(loader, device, z_dim=100, data_dim=29, num_classes=2, epochs=300,
               g_lr=2e-4, d_lr=5e-5,
               save_path="outputs/cgan.pt",
               plot_path="outputs/plots/cgan_loss.png"):
    G = Generator(z_dim, data_dim, num_classes).to(device)
    D = Discriminator(data_dim, num_classes).to(device)
    bce = nn.BCELoss()
    g_opt = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999))

    g_losses, d_losses = [], []

    for epoch in range(epochs):
        g_epoch, d_epoch = [], []
        for real, labels in loader:
            real = real.to(device)
            labels = labels.to(device)
            bs = real.size(0)
            real_lbl = torch.ones(bs, 1, device=device)
            fake_lbl = torch.zeros(bs, 1, device=device)

            # Train D
            z = torch.randn(bs, z_dim, device=device)
            fake = G(z, labels).detach()
            d_loss = bce(D(real, labels), real_lbl) + bce(D(fake, labels), fake_lbl)
            d_opt.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            d_opt.step()

            # Train G
            z = torch.randn(bs, z_dim, device=device)
            g_loss = bce(D(G(z, labels), labels), real_lbl)
            g_opt.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            g_opt.step()

            g_epoch.append(g_loss.item())
            d_epoch.append(d_loss.item())

        g_losses.append(sum(g_epoch) / len(g_epoch))
        d_losses.append(sum(d_epoch) / len(d_epoch))

        if (epoch + 1) % 50 == 0:
            print(f"[CGAN] Epoch {epoch+1}/{epochs}  "
                  f"D: {d_losses[-1]:.4f}  G: {g_losses[-1]:.4f}")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(G.state_dict(), save_path)
    print(f"[CGAN] Model saved to {save_path}")

    # Plot
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")
    ax.plot(g_losses, label="Generator", color="#A8C8E8", linewidth=2)
    ax.plot(d_losses, label="Discriminator", color="#F4A8A8", linewidth=2)
    ax.set_xlabel("Epoch", color="#555555")
    ax.set_ylabel("Loss", color="#555555")
    ax.set_title("CGAN Training Loss", color="#333333")
    ax.legend()
    ax.grid(True, color="#FFFFFF", linewidth=1.0)
    ax.tick_params(colors="#666666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    print(f"[CGAN] Loss curve saved to {plot_path}")

    return G, g_losses, d_losses


@torch.no_grad()
def generate_samples(G, n, z_dim, target_class, device):
    G.eval()
    z = torch.randn(n, z_dim, device=device)
    labels = torch.full((n,), target_class, dtype=torch.long, device=device)
    return G(z, labels).cpu()
