import argparse
import csv
import io
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn


class ChannelAttention(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        h = max(c // r, 4)
        self.mlp = nn.Sequential(nn.Linear(c, h), nn.ReLU(inplace=True), nn.Linear(h, c))

    def forward(self, x):
        a = self.mlp(F.adaptive_avg_pool2d(x, 1).flatten(1))
        m = self.mlp(F.adaptive_max_pool2d(x, 1).flatten(1))
        return x * torch.sigmoid(a + m)[:, :, None, None]


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2)

    def forward(self, x):
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.conv(s))


class Attn(nn.Module):
    def __init__(self, c, mode):
        super().__init__()
        self.ca = ChannelAttention(c) if mode in ("ca", "both") else nn.Identity()
        self.sa = SpatialAttention() if mode in ("sa", "both") else nn.Identity()

    def forward(self, x):
        return self.sa(self.ca(x))


def att(c, mode, use):
    return Attn(c, mode) if (mode != "none" and use) else nn.Identity()


def conv(i, o, s=1):
    return nn.Conv2d(i, o, 5, s, 2)


def deconv(i, o, s=1):
    return nn.ConvTranspose2d(i, o, 5, s, 2, output_padding=s - 1)


class DeepJSCC(nn.Module):
    def __init__(self, arch="brief", mode="both", place="enc", c_out=16):
        super().__init__()
        e = place in ("enc", "both")
        d = place in ("dec", "both")
        P = nn.PReLU
        if arch == "brief":
            enc = [conv(3, 32, 2), P(), conv(32, 64, 2), P(), conv(64, 64), P(), att(64, mode, e), conv(64, c_out)]
            dec = [deconv(c_out, 64), P(), att(64, mode, d), deconv(64, 32, 2), P(), deconv(32, 3, 2), nn.Sigmoid()]
        else:
            enc = [conv(3, 16, 2), P(), conv(16, 32, 2), P(), conv(32, 32), P(), conv(32, 32), P(), att(32, mode, e), conv(32, c_out)]
            dec = [deconv(c_out, 32), P(), deconv(32, 32), P(), deconv(32, 32), P(), att(32, mode, d), deconv(32, 16, 2), P(), deconv(16, 3, 2), nn.Sigmoid()]
        self.enc = nn.Sequential(*enc)
        self.dec = nn.Sequential(*dec)

    def forward(self, x, snr_db):
        z = self.enc(x)
        n = z[0].numel()
        z = z * math.sqrt(n) / z.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, 1, 1, 1)
        p = z.pow(2).flatten(1).mean(1).view(-1, 1, 1, 1)
        sigma = (p / (10 ** (snr_db / 10))).sqrt()
        return self.dec(z + torch.randn_like(z) * sigma)


def psnr(x, y):
    mse = F.mse_loss(y, x, reduction="none").flatten(1).mean(1).clamp_min(1e-10)
    return (10 * torch.log10(1.0 / mse)).mean().item()


@torch.no_grad()
def evaluate(model, loader, snr_db, dev, reps):
    model.eval()
    p = s = 0.0
    n = 0
    for x, _ in loader:
        x = x.to(dev)
        for _ in range(reps):
            y = model(x, snr_db)
            b = x.size(0)
            p += psnr(x, y) * b
            s += ssim_fn(y, x, data_range=1.0).item() * b
            n += b
    return p / n, s / n


@torch.no_grad()
def latency_ms(model, dev, runs=100):
    model.eval()
    x = torch.rand(1, 3, 32, 32, device=dev)
    for _ in range(20):
        model(x, 10.0)
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(runs):
        model(x, 10.0)
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / runs * 1000


def jpeg_bound(imgs, snr_db, kn):
    ps, ss = [], []
    cap = math.log2(1 + 10 ** (snr_db / 10))
    for x in imgs:
        n = x.numel()
        budget = kn * cap * n
        arr = (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        im = Image.fromarray(arr)
        lo, hi, best = 1, 95, None
        while lo <= hi:
            mid = (lo + hi) // 2
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=mid)
            if buf.tell() * 8 <= budget:
                best = buf.getvalue()
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None:
            rec = x.mean(dim=(1, 2), keepdim=True).expand_as(x).clone()
        else:
            rec = torch.from_numpy(np.array(Image.open(io.BytesIO(best)).convert("RGB"))).permute(2, 0, 1).float() / 255
        ps.append(psnr(x[None], rec[None]))
        ss.append(ssim_fn(rec[None], x[None], data_range=1.0).item())
    return float(np.mean(ps)), float(np.mean(ss))


@torch.no_grad()
def eval_kodak(model, folder, snrs, dev, reps):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    imgs = [T.ToTensor()(Image.open(os.path.join(folder, f)).convert("RGB")) for f in files]
    model.eval()
    out = []
    for snr in snrs:
        p = s = 0.0
        for x in imgs:
            x = x[None].to(dev)
            for _ in range(reps):
                y = model(x, float(snr))
                p += psnr(x, y)
                s += ssim_fn(y, x, data_range=1.0).item()
        out.append((snr, p / (len(imgs) * reps), s / (len(imgs) * reps)))
    return out


def train(model, loader, dev, epochs, lr, snr_train, tag):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(epochs * 0.7), int(epochs * 0.9)], 0.1)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, _ in loader:
            x = x.to(dev, non_blocking=True)
            loss = F.mse_loss(model(x, snr_train), x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        print(f"[{tag}] epoch {ep + 1}/{epochs} loss {tot / len(loader):.6f}", flush=True)


def plot(results, jpeg, snrs, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, r in results.items():
        ax[0].plot(snrs, r["psnr"], marker="o", label=name)
        ax[1].plot(snrs, r["ssim"], marker="o", label=name)
    if jpeg:
        ax[0].plot(snrs, jpeg["psnr"], "k--", marker="x", label="JPEG (capacity bound)")
        ax[1].plot(snrs, jpeg["ssim"], "k--", marker="x", label="JPEG (capacity bound)")
    for a, t in zip(ax, ["PSNR (dB)", "SSIM"]):
        a.set_xlabel("Channel SNR (dB)")
        a.set_ylabel(t)
        a.grid(alpha=0.3)
    ax[0].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=int, default=1)
    ap.add_argument("--arch", choices=["brief", "paper"], default="brief")
    ap.add_argument("--attn", choices=["none", "ca", "sa", "both"], default="both")
    ap.add_argument("--place", choices=["enc", "dec", "both"], default="both")
    ap.add_argument("--snr_train", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--jpeg_n", type=int, default=500)
    ap.add_argument("--kodak_dir", type=str, default="")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    tf = T.ToTensor()
    tr = torchvision.datasets.CIFAR10("./data", True, tf, download=True)
    te = torchvision.datasets.CIFAR10("./data", False, tf, download=True)
    trl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True)
    tel = DataLoader(te, 500, num_workers=a.workers, pin_memory=True)

    if a.suite:
        configs = [
            ("baseline", "none", "enc"),
            ("channel_only_enc", "ca", "enc"),
            ("spatial_only_enc", "sa", "enc"),
            ("cbam_enc", "both", "enc"),
            ("cbam_enc_dec", "both", "both"),
        ]
    else:
        configs = [(f"{a.attn}_{a.place}", a.attn, a.place)]

    snrs = list(range(0, 21, 2))
    results = {}
    summary = []

    for name, mode, place in configs:
        torch.manual_seed(a.seed)
        model = DeepJSCC(a.arch, mode, place).to(dev)
        params = sum(p.numel() for p in model.parameters())
        train(model, trl, dev, a.epochs, a.lr, a.snr_train, name)
        torch.save(model.state_dict(), os.path.join(a.out, f"{name}_{a.arch}_snr{int(a.snr_train)}.pt"))
        ps, ss = [], []
        for snr in snrs:
            p, s = evaluate(model, tel, float(snr), dev, a.reps)
            ps.append(p)
            ss.append(s)
            print(f"[{name}] snr {snr} psnr {p:.3f} ssim {s:.4f}", flush=True)
        lat = latency_ms(model, dev)
        results[name] = {"psnr": ps, "ssim": ss}
        summary.append((name, params, lat))
        if a.kodak_dir:
            k = eval_kodak(model, a.kodak_dir, snrs, dev, max(1, a.reps // 2))
            with open(os.path.join(a.out, f"kodak_{name}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["snr_db", "psnr", "ssim"])
                w.writerows(k)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    imgs = [te[i][0] for i in range(a.jpeg_n)]
    jp, js = [], []
    for snr in snrs:
        p, s = jpeg_bound(imgs, float(snr), 1 / 6)
        jp.append(p)
        js.append(s)
    jpeg = {"psnr": jp, "ssim": js}

    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "snr_db", "psnr", "ssim"])
        for name, r in results.items():
            for i, snr in enumerate(snrs):
                w.writerow([name, snr, f"{r['psnr'][i]:.4f}", f"{r['ssim'][i]:.5f}"])
        for i, snr in enumerate(snrs):
            w.writerow(["jpeg_capacity_bound", snr, f"{jp[i]:.4f}", f"{js[i]:.5f}"])

    with open(os.path.join(a.out, "complexity.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "params", "latency_ms_per_image"])
        for n, p, l in summary:
            w.writerow([n, p, f"{l:.3f}"])

    plot(results, jpeg, snrs, os.path.join(a.out, "results.png"))

    print("\nmodel,params,latency_ms")
    for n, p, l in summary:
        print(f"{n},{p},{l:.3f}")
    base = results.get("baseline")
    if base:
        print("\nmodel,mean_dPSNR_0-10dB,mean_dSSIM_0-10dB")
        idx = [i for i, s in enumerate(snrs) if s <= 10]
        for name, r in results.items():
            if name == "baseline":
                continue
            dp = np.mean([r["psnr"][i] - base["psnr"][i] for i in idx])
            ds = np.mean([r["ssim"][i] - base["ssim"][i] for i in idx])
            print(f"{name},{dp:+.3f},{ds:+.4f}")


if __name__ == "__main__":
    main()
