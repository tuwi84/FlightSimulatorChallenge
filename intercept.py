#!/usr/bin/env python3
# intercept.py —  AirSim/UE
# 
# How to use
#   python intercept.py --guidance-source ue --ue-target-name Target --speed-cap-xy 14 --speed-cap-z 6 --capture-dist 8 --rel-kp 1.2 --rel-kd 0.8 --rv 0.8 --rt 0.8 --debug 0.3 --wind "0,0,0" --noise-pos 0.0 --noise-vel 0.0

import cosysairsim as airsim
import time, math, argparse, os, csv, json, sys, subprocess, random
import numpy as np
from typing import Optional, Tuple
from datetime import datetime

PORT = 41451
DT   = 0.05




# ---------------- utilities ----------------
def flush_traces(c):
    """Clear any persistent debug lines/markers regardless of AirSim build."""
    try:
        c.simFlushPersistentLines()
    except Exception:
        pass
    try:
        c.simFlushPersistentMarkers()
    except Exception:
        pass

def clamp_xy(vxy: np.ndarray, vmax: float) -> np.ndarray:
    n = float(np.linalg.norm(vxy))
    return vxy if n <= 1e-9 or n <= vmax else vxy * (vmax / n)

def kin_state(c, name) -> Tuple[np.ndarray, np.ndarray]:
    k = c.getMultirotorState(vehicle_name=name).kinematics_estimated
    p = np.array([k.position.x_val, k.position.y_val, k.position.z_val], float)
    v = np.array([k.linear_velocity.x_val, k.linear_velocity.y_val, k.linear_velocity.z_val], float)
    return p, v

def resolve_scene_key(c, base_name: str) -> Optional[str]:
    import re
    exact = c.simListSceneObjects(f"^{re.escape(base_name)}$")
    if exact:
        print(f"[ue] resolved exactly: {exact[0]}")
        return exact[0]
    names = c.simListSceneObjects(".*")
    if not names:
        print("[ue] no scene objects returned"); return None
    cand = [n for n in names if base_name.lower() in n.lower()]
    if not cand:
        print(f"[ue] no scene object contains '{base_name}'"); return None
    cand.sort(key=lambda s: (("." in s), len(s)))
    pick = cand[0]
    print(f"[ue] pick '{pick}'"); return pick

def get_scene_pos(c, key: Optional[str]) -> Optional[np.ndarray]:
    if key is None: return None
    try:
        pose = c.simGetObjectPose(key)
        return np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val], float)
    except Exception:
        return None

def make_run_dir(root: str, prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(root, f"{prefix}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def parse_vec3(s: str) -> np.ndarray:
    try:
        parts = [float(x.strip()) for x in s.replace(";",",").split(",") if x.strip()!=""]
        if len(parts) != 3: return np.zeros(3, float)
        return np.array(parts, float)
    except Exception:
        return np.zeros(3, float)




# --------- target velocity estimator (for UE) ---------
class VelEstimator:
    def __init__(self, ema=0.5):
        self.prev = None
        self.vhat = np.zeros(3, float)
        self.ema  = float(ema)
        self.ptime = None

    def update(self, p: np.ndarray, now: float) -> np.ndarray:
        if self.prev is None or self.ptime is None:
            self.prev = p.copy(); self.ptime = now; return self.vhat.copy()
        dt = max(1e-3, now - self.ptime)
        v  = (p - self.prev) / dt
        self.vhat = (1 - self.ema) * self.vhat + self.ema * v
        self.prev = p.copy(); self.ptime = now
        return self.vhat.copy()



# ---------------- CSV / metrics helpers ----------------
def open_csv(run_dir: str):
    path = os.path.join(run_dir, "traj.csv")
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow([
        "t",
        "S_as_x","S_as_y","S_as_z",
        "S_as_vx","S_as_vy","S_as_vz",
        "S_meas_x","S_meas_y","S_meas_z",
        "S_meas_vx","S_meas_vy","S_meas_vz",
        "T_x","T_y","T_z",
        "T_vx","T_vy","T_vz",
        "dist_ue",
        "mode",
        "rdot"
    ])
    return f, w, path

def write_metrics(run_dir: str, csv_path: str, metrics: dict):
    jpath = os.path.join(run_dir, "metrics.json")
    m = dict(metrics)
    m["run_dir"] = os.path.abspath(run_dir)
    m["csv"] = os.path.abspath(csv_path) if csv_path else ""
    try:
        with open(jpath, "w", encoding="utf-8") as jf:
            json.dump(m, jf, indent=2)
        print(f"[log] metrics -> {jpath}")
    except Exception as e:
        print(f"[log] metrics write failed: {e}")

def try_plot(csv_path: str):
    """Best-effort: call plot_csv.py to generate PNGs next to traj.csv"""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(here, "plot_csv.py"), "plot_csv.py"):
        try:
            subprocess.check_call([sys.executable, candidate, csv_path])
            return True
        except Exception:
            pass
    print("[plot] plot_csv.py not found or failed — skipping.")
    return False



# -------------- main --------------
def main():
    ap = argparse.ArgumentParser(description="Intercept with velocity-matching capture + CSV/UE traces/wind/noise")

    # Pose sources
    ap.add_argument("--guidance-source", choices=["ue","airsim"], default="ue")
    ap.add_argument("--ue-target-name", type=str, default="Target")
    ap.add_argument("--ue-drone-name",  type=str, default="Drone1")
    ap.add_argument("--as-target-name", type=str, default="Target")
    ap.add_argument("--as-drone-name",  type=str, default="Drone1")

    # Speed caps
    ap.add_argument("--speed-cap-xy", type=float, default=14.0)
    ap.add_argument("--speed-cap-z",  type=float, default=6.0)

    # Far-range pursuit (lead PD)
    ap.add_argument("--kp", type=float, default=1.2)
    ap.add_argument("--kd", type=float, default=0.35)
    ap.add_argument("--tau-max", type=float, default=2.0, help="lead lookahead cap [s]")

    # Capture (near/overshoot)
    ap.add_argument("--capture-dist", type=float, default=8.0, help="enter capture when dist <= this [m]")
    ap.add_argument("--rel-kp", type=float, default=1.2, help="relative position gain")
    ap.add_argument("--rel-kd", type=float, default=0.8, help="relative velocity gain")
    ap.add_argument("--always-capture", action="store_true", help="force capture law at all ranges (debug)")

    # Target velocity estimate (UE mode)
    ap.add_argument("--vt-ema", type=float, default=0.5, help="EMA for UE velocity estimate 0..1")

    # Hit test (UE distance)
    ap.add_argument("--rv", type=float, default=0.8)
    ap.add_argument("--rt", type=float, default=0.8)

    # Traces in UE + replay export
    ap.add_argument("--trace-every", type=int, default=5, help="plot every N steps to reduce overhead (line strips)")
    ap.add_argument("--trace-duration", type=float, default=0.0, help="seconds; 0 = persistent until reset")
    ap.add_argument("--hit-pause", type=float, default=2.0, help="seconds to pause after HIT before reset")

    # Logging
    ap.add_argument("--log-dir", type=str, default="logs")
    ap.add_argument("--csv-prefix", type=str, default="run")
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--write-metrics-json", action="store_true", default=True)

    # Wind & measurement noise
    ap.add_argument("--wind", type=str, default="0,0,0", help='constant wind velocity "wx,wy,wz" [m/s] (NED)')
    ap.add_argument("--noise-pos", type=float, default=0.0, help="stddev of Gaussian noise on measured positions [m]")
    ap.add_argument("--noise-vel", type=float, default=0.0, help="stddev of Gaussian noise on measured velocities [m/s]")
    ap.add_argument("--wind-applies-to", choices=["meas","none"], default="meas",
                    help="apply wind bias to measured velocities (simple airspeed vs ground)")

    # Misc
    ap.add_argument("--tmax", type=float, default=60.0)
    ap.add_argument("--debug", type=float, default=0.3)
    args = ap.parse_args()

    # Prepare run folder
    run_dir = make_run_dir(args.log_dir, args.csv_prefix)
    print(f"[log] run dir -> {run_dir}")

    c = airsim.MultirotorClient(ip="127.0.0.1", port=PORT)
    c.confirmConnection(); print("Connected.")

    # Clear any old persistent debug lines from previous sessions
    flush_traces(c)

    # Resolve UE keys
    ue_target_key = resolve_scene_key(c, args.ue_target_name) if args.guidance_source == "ue" else None
    ue_drone_key  = resolve_scene_key(c, args.ue_drone_name)

    # Arm / takeoff (AirSim vehicles)
    for n in [args.as_drone_name, args.as_target_name]:
        try:
            c.enableApiControl(True, n); c.armDisarm(True, n)
        except Exception:
            pass
    try:
        c.takeoffAsync(vehicle_name=args.as_drone_name).join()
        c.takeoffAsync(vehicle_name=args.as_target_name).join()
    except Exception:
        pass

    # CSV setup
    csv_file = None; csv_writer = None; csv_path = ""
    if not args.no_csv:
        csv_file, csv_writer, csv_path = open_csv(run_dir)
        print(f"[log] csv -> {csv_path}")

    # Estimator for UE target velocity
    vtest = VelEstimator(ema=args.vt_ema)

    # Wind/noise setup
    wind = parse_vec3(args.wind)
    npos = float(args.noise_pos)
    nvel = float(args.noise_vel)
    rng  = np.random.default_rng()

    def add_noise_pos(p: np.ndarray) -> np.ndarray:
        if npos <= 0: return p
        return p + rng.normal(0.0, npos, size=3)

    def add_noise_vel(v: np.ndarray) -> np.ndarray:
        if nvel <= 0: return v
        return v + rng.normal(0.0, nvel, size=3)

    # Trace collections (for on-screen) and replay export
    drone_strip = []
    target_strip = []
    replay_drone = []   # list of {"t", "x","y","z"}
    replay_target = []

    t0 = time.time(); next_dbg = 0.0
    hit = False; min_dist = 1e9
    last_mode = "PURSUIT"

    try:
        while True:
            now = time.time()
            t = now - t0
            if t > args.tmax:
                break

            # Drone state (AirSim truth for dynamics/logs)
            pS_as, vS_as = kin_state(c, args.as_drone_name)

            # Target pose & velocity (UE or AirSim)
            if args.guidance_source == "ue":
                pT_ue = get_scene_pos(c, ue_target_key)
                if pT_ue is None:
                    pT_as, vT_as = kin_state(c, args.as_target_name)
                    pT, vT = pT_as, vT_as
                else:
                    pT = pT_ue
                    vT = vtest.update(pT, now)
            else:
                pT, vT = kin_state(c, args.as_target_name)

            # Drone pose from UE for distance truth
            pS_ue_tmp = get_scene_pos(c, ue_drone_key)
            pS_ue = pS_ue_tmp if pS_ue_tmp is not None else pS_as

            # Measurements (apply noise, and optionally wind bias on velocities)
            pS_meas = add_noise_pos(pS_as.copy())
            vS_meas = add_noise_vel(vS_as.copy())
            pT_meas = add_noise_pos(pT.copy())
            vT_meas = add_noise_vel(vT.copy())
            if args.wind_applies_to == "meas":
                vS_meas = vS_meas - wind
                vT_meas = vT_meas - wind

            # Distance for HIT (UE truth)
            dist = float(np.linalg.norm(pT - pS_ue))
            min_dist = min(min_dist, dist)

            # Relative (use measured states for guidance)
            rel_p = pT_meas - pS_meas
            rel_v = vT_meas - vS_meas
            r = float(np.linalg.norm(rel_p))
            r_hat = rel_p / (r + 1e-9)
            rdot = float(np.dot(r_hat, rel_v))  # positive when opening

            # Mode selection
            enter_capture = args.always_capture or (r <= args.capture_dist) or (r <= 1.25*args.capture_dist and rdot > 0.0)
            if enter_capture:
                v_cmd = vT_meas + args.rel_kp * rel_p - args.rel_kd * (vS_meas - vT_meas)
                mode = "CAPTURE"
            else:
                v_rel = rel_v
                vrel2 = float(np.dot(v_rel, v_rel)) + 1e-9
                tau = max(0.0, min(args.tau_max, float(-np.dot(rel_p, v_rel) / vrel2)))
                lead = pT_meas + tau * vT_meas
                err  = lead - pS_meas
                v_cmd = args.kp * err - args.kd * vS_meas
                mode = "PURSUIT"
            last_mode = mode

            # Apply caps
            vxy = clamp_xy(v_cmd[:2], args.speed_cap_xy)
            vz  = max(-args.speed_cap_z, min(args.speed_cap_z, float(v_cmd[2])))
            v_cmd = np.array([vxy[0], vxy[1], vz], float)

            # Send command (ground-referenced; AirSim handles physics)
            c.moveByVelocityAsync(float(v_cmd[0]), float(v_cmd[1]), float(v_cmd[2]),
                                  DT,
                                  drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                  yaw_mode=airsim.YawMode(False, 0.0),
                                  vehicle_name=args.as_drone_name)

            # Log row
            if csv_writer is not None:
                csv_writer.writerow([
                    f"{t:.3f}",
                    f"{pS_as[0]:.3f}", f"{pS_as[1]:.3f}", f"{pS_as[2]:.3f}",
                    f"{vS_as[0]:.3f}", f"{vS_as[1]:.3f}", f"{vS_as[2]:.3f}",
                    f"{pS_meas[0]:.3f}", f"{pS_meas[1]:.3f}", f"{pS_meas[2]:.3f}",
                    f"{vS_meas[0]:.3f}", f"{vS_meas[1]:.3f}", f"{vS_meas[2]:.3f}",
                    f"{pT[0]:.3f}",    f"{pT[1]:.3f}",    f"{pT[2]:.3f}",
                    f"{vT[0]:.3f}",    f"{vT[1]:.3f}",    f"{vT[2]:.3f}",
                    f"{dist:.3f}",
                    mode,
                    f"{rdot:.3f}",
                ])

            # --- traces (UE) + replay collection ---
            replay_drone.append({"t": round(t,3), "x": float(pS_ue[0]), "y": float(pS_ue[1]), "z": float(pS_ue[2])})
            replay_target.append({"t": round(t,3), "x": float(pT[0]),   "y": float(pT[1]),   "z": float(pT[2])})

            # Grow continuous line strips; redraw every N steps. Colors are 0..1 floats.
            try:
                drone_strip.append(airsim.Vector3r(float(pS_ue[0]), float(pS_ue[1]), float(pS_ue[2])))
                target_strip.append(airsim.Vector3r(float(pT[0]),   float(pT[1]),   float(pT[2])))
                if (len(drone_strip) % max(1, args.trace_every)) == 0:
                    c.simPlotLineStrip(drone_strip, color_rgba=[0.0, 0.4, 1.0, 1.0], thickness=6.0, duration=args.trace_duration, is_persistent=True)
                    c.simPlotLineStrip(target_strip, color_rgba=[1.0, 0.1, 0.1, 1.0], thickness=6.0, duration=args.trace_duration, is_persistent=True)
            except Exception:
                pass

            # HIT (UE distance)
            if dist <= (args.rv + args.rt):
                print(f"HIT  t={t:.2f}s  dist={dist:.3f}  mode={mode}")
                hit = True
                # clear lines so next run starts clean
                flush_traces(c)
                # small pause before resetting (for visualization/logging)
                time.sleep(max(0.0, args.hit_pause))
                try:
                    print("[reset] Resetting AirSim world (same as Backspace in UE)")
                    c.reset()
                    time.sleep(0.5)  # allow sim to settle
                except Exception as e:
                    print("[reset] Reset failed:", e)
                break

            # Debug
            if args.debug > 0 and (now - t0) >= next_dbg:
                next_dbg += args.debug
                print(f"t={t:5.2f}  dist={dist:6.2f}  mode={mode:8s}  rdot={rdot:+5.2f}  "
                      f"|v|=({np.linalg.norm(v_cmd[:2]):4.1f},{abs(v_cmd[2]):4.1f})  "
                      f"min={min_dist:5.2f}")

            time.sleep(DT)

    finally:
        try:
            for n in [args.as_drone_name, args.as_target_name]:
                c.hoverAsync(vehicle_name=n).join()
                c.landAsync(vehicle_name=n).join()
                c.armDisarm(False, n); c.enableApiControl(False, n)
        except Exception:
            pass
        if csv_file is not None:
            try: csv_file.flush(); csv_file.close()
            except Exception: pass

    # Metrics
    metrics = {
        "hit": bool(hit),
        "t_end": round(time.time()-t0, 3),
        "min_dist": round(min_dist, 3),
        "guidance_source": args.guidance_source,
        "mode_final": last_mode,
        "capture_dist": args.capture_dist,
        "rv": args.rv, "rt": args.rt,
        "speed_cap_xy": args.speed_cap_xy, "speed_cap_z": args.speed_cap_z,
        "kp": args.kp, "kd": args.kd, "rel_kp": args.rel_kp, "rel_kd": args.rel_kd,
        "tau_max": args.tau_max,
        "wind": list(map(float, parse_vec3(args.wind))),
        "noise_pos": args.noise_pos, "noise_vel": args.noise_vel
    }
    print("[metrics]", metrics)

    # Write metrics + replay + plots
    if args.write_metrics_json:
        write_metrics(run_dir, csv_path, metrics)

    # replay.json
    replay = {"drone": replay_drone, "target": replay_target}
    try:
        rpath = os.path.join(run_dir, "replay.json")
        with open(rpath, "w", encoding="utf-8") as jf:
            json.dump(replay, jf, separators=(",",":"))
        print(f"[log] replay  -> {rpath}")
    except Exception as e:
        print("[log] replay write failed:", e)

    if csv_path:
        if try_plot(csv_path):
            print("[plot] plots written next to CSV")

    print("\n=== RUN COMPLETE ===")
    print(f"Folder: {os.path.abspath(run_dir)}")
    print(f"Hit: {metrics['hit']}  MinDist: {metrics['min_dist']}  t_end: {metrics['t_end']}  FinalMode: {metrics['mode_final']}")

if __name__ == "__main__":
    main()

