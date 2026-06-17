"""Generate k-Wave FMC cubes on the GPU (server kwave310 env).

Side-drilled holes are modelled as finite air-filled disks in a steel matrix, so
the scattering has real physics (finite size, directivity, creeping waves) that the
analytic point-scatterer steering dictionary CANNOT represent. Output cubes match
the analytic pipeline's convention: A[tx, rx, t], resampled to (fs_out, n_t_out),
saved as .npz with the true defect positions for the frontier analysis.

Run:  D:\\Anaconda3\\envs\\kwave310\\python.exe kwave_fmc_gen.py --out C:\\Users\\sheng\\kwave_data --n 6
"""
import argparse
import os
import time

import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.signals import tone_burst

# ----- fixed acquisition geometry (matches configs/default.yaml) -----
N = 64
PITCH = 0.5e-3
C0 = 6300.0
RHO0 = 7800.0
F0 = 5.0e6
N_CYCLES = 3
FS_OUT = 50.0e6
N_T_OUT = 1024
DX = 0.25e-3                      # PPW = lambda/dx = 1.26mm/0.25 ~ 5; pitch = 2 grid pts
RHO_DEF = 1000.0                 # SDH = strong density-contrast disk (R~-0.77); c kept
                                 # homogeneous so the low-Z region stays well-sampled/stable,
                                 # while finite size (ka~3) still breaks the point monopole model.
T_END = N_T_OUT / FS_OUT         # 20.48 us


def build_grid():
    pitch_pts = int(round(PITCH / DX))            # 2
    span = (N - 1) * pitch_pts                    # 126
    margin_lat = 32
    z_surf = 8
    depth_pts = int(round(40e-3 / DX))            # 160
    Ny = span + 2 * margin_lat
    Nx = z_surf + depth_pts + 20
    col0 = (Ny - span) // 2
    elem_cols = col0 + np.arange(N) * pitch_pts   # ascending => element order
    col_center = (Ny - 1) / 2.0
    return dict(Nx=Nx, Ny=Ny, z_surf=z_surf, elem_cols=elem_cols, col_center=col_center)


def make_medium(geom, defects):
    """Homogeneous steel sound speed; finite low-density disks at the defect positions."""
    Nx, Ny = geom["Nx"], geom["Ny"]
    rho = np.full((Nx, Ny), RHO0, dtype=np.float32)
    rr, cc = np.mgrid[0:Nx, 0:Ny]
    for (x, z, radius) in defects:
        col = geom["col_center"] + x / DX
        row = geom["z_surf"] + z / DX
        rad_pts = max(radius / DX, 1.0)
        disk = (rr - row) ** 2 + (cc - col) ** 2 <= rad_pts ** 2
        rho[disk] = RHO_DEF
    return rho


def sample_cracks(rng, n):
    """n thin, tilted, rough cracks — specular + diffuse, strongly angle-dependent,
    so the analytic isotropic-monopole steering model is mis-specified."""
    x_range, z_range = (-8e-3, 8e-3), (12e-3, 30e-3)
    cracks = []
    for _ in range(n):
        L = float(rng.uniform(2e-3, 6e-3))
        tilt = float(rng.uniform(-np.pi / 3, np.pi / 3))     # +-60 deg from horizontal
        thick = float(rng.uniform(0.2e-3, 0.4e-3))
        cx = float(rng.uniform(x_range[0] + L / 2, x_range[1] - L / 2))
        cz = float(rng.uniform(z_range[0] + L / 2, z_range[1] - L / 2))
        rfreq = float(rng.uniform(2, 5) / L)                  # rough-face spatial freq
        rphase = float(rng.uniform(0, 2 * np.pi))
        cracks.append((cx, cz, L, tilt, thick, rfreq, rphase))
    return cracks


def make_medium_cracks(geom, cracks):
    Nx, Ny = geom["Nx"], geom["Ny"]
    rho = np.full((Nx, Ny), RHO0, dtype=np.float32)
    rr, cc = np.mgrid[0:Nx, 0:Ny]
    X = (cc - geom["col_center"]) * DX
    Z = (rr - geom["z_surf"]) * DX
    for (cx, cz, L, tilt, thick, rfreq, rphase) in cracks:
        u = (X - cx) * np.cos(tilt) + (Z - cz) * np.sin(tilt)
        v = -(X - cx) * np.sin(tilt) + (Z - cz) * np.cos(tilt)
        half_t = 0.5 * thick * (1.0 + 0.6 * np.sin(2 * np.pi * rfreq * u + rphase))  # rough faces
        rho[(np.abs(u) <= L / 2) & (np.abs(v) <= np.abs(half_t))] = RHO_DEF
    return rho


def crack_points(cracks, n_pts=5):
    """Points along each crack centerline — the fairest analytic line-of-monopoles model."""
    pts = []
    for (cx, cz, L, tilt, *_rest) in cracks:
        for t in np.linspace(-L / 2, L / 2, n_pts):
            pts.append((float(cx + t * np.cos(tilt)), float(cz + t * np.sin(tilt)), 1.0))
    return pts


def run_fmc(geom, rho, *, gpu=True):
    Nx, Ny = geom["Nx"], geom["Ny"]
    elem_cols = geom["elem_cols"].astype(int)
    z_surf = geom["z_surf"]                         # rho: prebuilt density map, copied per sim

    kg0 = kWaveGrid([Nx, Ny], [DX, DX])
    kg0.makeTime(C0, t_end=T_END)
    dt = kg0.dt
    t_kwave = np.arange(int(kg0.Nt)) * dt
    t_out = np.arange(N_T_OUT) / FS_OUT
    sig = tone_burst(1.0 / dt, F0, N_CYCLES)

    cube = np.zeros((N, N, N_T_OUT), dtype=np.float32)
    for i in range(N):
        # fresh objects each transmit: pml_inside=False expands kgrid in place, so
        # reusing kgrid/sensor across sims corrupts the binary-mask shape check.
        kgrid = kWaveGrid([Nx, Ny], [DX, DX])
        kgrid.makeTime(C0, t_end=T_END)
        medium = kWaveMedium(sound_speed=C0, density=rho.copy())
        sensor = kSensor()
        smask = np.zeros((Nx, Ny)); smask[z_surf, elem_cols] = 1
        sensor.mask = smask
        sensor.record = ["p"]
        source = kSource()
        pmask = np.zeros((Nx, Ny)); pmask[z_surf, elem_cols[i]] = 1
        source.p_mask = pmask
        source.p = sig
        sopt = SimulationOptions(pml_inside=False, save_to_disk=True, data_cast="single")
        eopt = SimulationExecutionOptions(is_gpu_simulation=gpu, delete_data=True)
        out = kspaceFirstOrder2D(kgrid=kgrid, medium=medium, source=source,
                                 sensor=sensor, simulation_options=sopt,
                                 execution_options=eopt)
        p = np.asarray(out["p"])                 # (nt, N) in ascending-column order
        if p.shape[0] == N and p.shape[1] != N:  # guard against (N, nt)
            p = p.T
        for j in range(N):
            cube[i, j] = np.interp(t_out, t_kwave, p[:, j])
    return cube


def sample_defects(rng, n_def):
    """n_def finite SDHs, radius ~0.6 mm, pairwise sep >= 1.3*lambda."""
    lam = C0 / F0
    min_sep = 1.3 * lam
    x_range, z_range = (-8e-3, 8e-3), (12e-3, 30e-3)
    pts = []
    tries = 0
    while len(pts) < n_def and tries < 5000:
        tries += 1
        x = float(rng.uniform(*x_range)); z = float(rng.uniform(*z_range))
        if all((x - px) ** 2 + (z - pz) ** 2 >= min_sep ** 2 for px, pz, _ in pts):
            pts.append((x, z, float(rng.uniform(0.5e-3, 0.7e-3))))  # radius
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kwave_data")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="first phantom id (for resume/sharding)")
    ap.add_argument("--max-ndef", type=int, default=8)
    ap.add_argument("--ndefs", default="", help="comma list of fixed n_def (probe mode); else random")
    ap.add_argument("--defect", default="disk", choices=["disk", "crack"])
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    geom = build_grid()
    print(f"grid Nx={geom['Nx']} Ny={geom['Ny']} (dx={DX*1e3}mm); {N} elements", flush=True)

    # Independent random configs (phantom-level split = no (tx,rx) leakage). The
    # seed is offset by phantom id so --start can resume/shard deterministically.
    fixed = [int(x) for x in args.ndefs.split(",") if x] if args.ndefs else None
    for k in range(args.n):
        pid = args.start + k
        rng = np.random.default_rng(args.seed + pid)
        if args.defect == "crack":
            nd = int(rng.integers(1, 5))                       # 1-4 cracks
        else:
            nd = fixed[k % len(fixed)] if fixed else int(rng.integers(1, args.max_ndef + 1))
        path = os.path.join(args.out, f"kw_{pid:04d}_nd{nd}.npz")
        if os.path.exists(path):                               # resume: skip done
            print(f"[{pid}] exists, skip", flush=True)
            continue
        if args.defect == "crack":
            cracks = sample_cracks(rng, nd)
            rho = make_medium_cracks(geom, cracks)
            defects = crack_points(cracks)                     # line-of-monopoles for analytic baseline
            extra = {"cracks": np.array(cracks)}
        else:
            defects = sample_defects(rng, nd)
            rho = make_medium(geom, defects)
            extra = {}
        t0 = time.time()
        cube = run_fmc(geom, rho, gpu=not args.cpu)
        np.savez_compressed(path, cube=cube.astype(np.float32),
                            defects=np.array([(x, z, r) for x, z, r in defects]),
                            c0=C0, f0=F0, fs=FS_OUT, n_t=N_T_OUT, pitch=PITCH, n_elements=N, **extra)
        print(f"[{pid}] {args.defect} nd={nd} -> {os.path.basename(path)}  ({time.time()-t0:.1f}s, "
              f"max {np.abs(cube).max():.2e})", flush=True)
    print("KWAVE_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
