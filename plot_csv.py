#!/usr/bin/env python3
import csv, sys, os, math
import numpy as np

# Headless / CI safe import
try:
    import matplotlib
    matplotlib.use("Agg")  # no GUI backend
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("[plot] matplotlib not installed — skipping plots.")
    plt = None

def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows.extend(r)
    return rows

def to_f(row, key, default=np.nan):
    try:
        return float(row[key])
    except Exception:
        return default

def main():
    if len(sys.argv) < 2:
        print("usage: plot_csv.py <traj.csv>")
        sys.exit(1)

    path = sys.argv[1]
    rows = load_csv(path)
    if not rows:
        print("empty csv")
        sys.exit(2)

    # === Extract columns ===
    t = np.array([to_f(r, "t") for r in rows])
    dist = np.array([to_f(r, "dist_ue") for r in rows])
    mode = [r.get("mode","") for r in rows]

    Sx = np.array([to_f(r,"S_as_x") for r in rows])
    Sy = np.array([to_f(r,"S_as_y") for r in rows])
    Sz = np.array([to_f(r,"S_as_z") for r in rows])
    Svx = np.array([to_f(r,"S_as_vx") for r in rows])
    Svy = np.array([to_f(r,"S_as_vy") for r in rows])
    Svz = np.array([to_f(r,"S_as_vz") for r in rows])
    Spd = np.sqrt(Svx**2 + Svy**2 + Svz**2)

    Tx = np.array([to_f(r,"T_x") for r in rows])
    Ty = np.array([to_f(r,"T_y") for r in rows])

    # detect mode switches
    marks = [0]
    for i in range(1, len(mode)):
        if mode[i] != mode[i-1]:
            marks.append(i)

    stem = os.path.join(os.path.dirname(path), "traj")

    if plt is None:
        print("[plot] matplotlib unavailable, no PNGs written.")
    else:
        # 1) Distance vs time
        plt.figure()
        plt.plot(t, dist, label="UE distance")
        for i in marks: plt.axvline(t[i], alpha=0.2)
        plt.xlabel("time [s]"); plt.ylabel("distance [m]")
        plt.title("Distance to target"); plt.legend(); plt.tight_layout()
        out1 = stem + "_dist.png"; plt.savefig(out1, dpi=150); print("plot ->", out1)

        # 2) Speed vs time
        plt.figure()
        plt.plot(t, Spd, label="Drone speed (AS)")
        for i in marks: plt.axvline(t[i], alpha=0.2)
        plt.xlabel("time [s]"); plt.ylabel("speed [m/s]")
        plt.title("Drone speed"); plt.legend(); plt.tight_layout()
        out2 = stem + "_speed.png"; plt.savefig(out2, dpi=150); print("plot ->", out2)

        # 3) XY top-down
        plt.figure()
        plt.plot(Tx, Ty, label="Target path")
        plt.plot(Sx, Sy, label="Drone path")
        plt.scatter([Sx[0]], [Sy[0]], marker="o", label="Start")
        plt.axis("equal"); plt.xlabel("X [m]"); plt.ylabel("Y [m]")
        plt.title("Top-down paths"); plt.legend(); plt.tight_layout()
        out3 = stem + "_xy.png"; plt.savefig(out3, dpi=150); print("plot ->", out3)

    # Console summary
    mind = float(np.nanmin(dist))
    t_hit = next((float(t[i]) for i,d in enumerate(dist) if d <= 1.6), None)
    print({"csv": os.path.basename(path),
           "min_dist": round(mind,3),
           "t_hit": (round(t_hit,3) if t_hit else None)})

if __name__ == "__main__":
    main()
