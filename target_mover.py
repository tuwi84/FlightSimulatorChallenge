#!/usr/bin/env python3
"""
target_mover.py — Drive a target along simple paths for AirSim.

Patterns
- circle | line | lemniscate

Modes
- airsim  control an AirSim multirotor 
- ue   UE actor along the path (simSetObjectPose)

Stop-on-hit
- --stop-on-hit ue|airsim|none  : compute spherical hit (rv+rt) using UE actor poses or AirSim vehicle states
- When a hit is detected (after --hit-cooldown seconds), the script stops the loop,
  hovers (airsim mode), prints one line, and exits(0).

Example (your run)
python target_mover.py --mode airsim --veh Target --pattern circle --radius 20 --omega 0.35 --z -12 \
  --speed-cap 12 --kp 1.0 --kd 0.3 --place-on-start \
  --stop-on-hit ue --hit-drone-ue Drone1 --hit-target-ue Target --hit-rv 0.8 --hit-rt 0.8
"""

import sys, time, math, argparse
from typing import Optional, Tuple
import numpy as np

import cosysairsim as airsim

PORT = 41451
DT   = 0.05  # main step [s]

# -------------------- utils --------------------
def clamp_norm(v: np.ndarray, vmax: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-9 or n <= vmax: return v
    return v * (vmax / n)

def kin_state(c: airsim.MultirotorClient, name: str) -> Tuple[np.ndarray, np.ndarray]:
    k = c.getMultirotorState(vehicle_name=name).kinematics_estimated
    p = np.array([k.position.x_val, k.position.y_val, k.position.z_val], float)
    v = np.array([k.linear_velocity.x_val, k.linear_velocity.y_val, k.linear_velocity.z_val], float)
    return p, v

def resolve_scene_key(c: airsim.MultirotorClient, base_name: str) -> Optional[str]:
    """Try exact, else best substring match."""
    import re
    exact = c.simListSceneObjects(f"^{re.escape(base_name)}$")
    if exact:
        print(f"[ue] resolved exactly: {exact[0]}")
        return exact[0]
    names = c.simListSceneObjects(".*") or []
    cand = [n for n in names if base_name.lower() in n.lower()]
    if not cand:
        print(f"[ue] no scene object contains '{base_name}'"); return None
    cand.sort(key=lambda s: (("." in s), len(s)))
    pick = cand[0]
    print(f"[ue] pick '{pick}'"); return pick

def get_scene_pos(c: airsim.MultirotorClient, key: str) -> Optional[np.ndarray]:
    try:
        pose = c.simGetObjectPose(key)
        return np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val], float)
    except Exception:
        return None

# -------------------- patterns (pos, vel) --------------------
def pat_circle(t: float, *, cx=0.0, cy=0.0, r=15.0, omega=0.12, z=-15.0):
    ang = omega * t
    pos = np.array([cx + r*math.cos(ang), cy + r*math.sin(ang), z], float)
    vel = np.array([-r*omega*math.sin(ang), r*omega*math.cos(ang), 0.0], float)
    return pos, vel

def pat_line(t: float, *, p0=(0,0,-15), p1=(30,0,-15), T=20.0):
    u = (t % (2*T))
    if u <= T:
        a = u / T;  pA, pB = np.array(p0,float), np.array(p1,float)
    else:
        a = (u - T) / T;  pA, pB = np.array(p1,float), np.array(p0,float)
    pos = (1-a)*pA + a*pB
    vel = (pB - pA) / T
    return pos, vel

def pat_lemniscate(t: float, *, cx=0.0, cy=0.0, z=-15.0, a=12.0, omega=0.15):
    th = omega * t
    denom = 1 + math.sin(th)**2
    x = a * math.cos(th) / denom
    y = a * math.sin(th) * math.cos(th) / denom
    # finite-diff velocity
    dt = 1e-2
    th2 = omega * (t + dt)
    denom2 = 1 + math.sin(th2)**2
    x2 = a * math.cos(th2) / denom2
    y2 = a * math.sin(th2) * math.cos(th2) / denom2
    vx = (x2 - x) / dt; vy = (y2 - y) / dt
    pos = np.array([cx + x, cy + y, z], float)
    vel = np.array([vx, vy, 0.0], float)
    return pos, vel

PATTERNS = {"circle": pat_circle, "line": pat_line, "lemniscate": pat_lemniscate}

# -------------------- hit checks --------------------
def hit_margin_ue(c, drone_key: Optional[str], target_key: Optional[str], rv: float, rt: float) -> Optional[float]:
    if not drone_key or not target_key: return None
    pD = get_scene_pos(c, drone_key); pT = get_scene_pos(c, target_key)
    if pD is None or pT is None: return None
    return float(np.linalg.norm(pD - pT)) - float(rv + rt)

def hit_margin_airsim(c, drone_veh: str, target_veh: str, rv: float, rt: float) -> Optional[float]:
    try:
        pD, _ = kin_state(c, drone_veh); pT, _ = kin_state(c, target_veh)
        return float(np.linalg.norm(pD - pT)) - float(rv + rt)
    except Exception:
        return None


# -------------------- movers --------------------
def move_target_airsim(c, veh_name, gen, params, speed_cap, kp, kd,
                       place_on_start, tmax, debug,
                       stop_mode, hit_drone_ue_key, hit_target_ue_key,
                       hit_drone_veh, hit_target_veh, hit_rv, hit_rt, hit_cooldown):
    t0 = time.time(); next_dbg = 0.0
    p0, _ = gen(0.0, **params)
    if place_on_start:
        c.moveToPositionAsync(float(p0[0]), float(p0[1]), float(p0[2]), 5.0,
                              vehicle_name=veh_name).join()
    try:
        while True:
            t = time.time() - t0
            if t > tmax: break

            pref, vref = gen(t, **params)
            p, v = kin_state(c, veh_name)
            v_cmd = kp * (pref - p) + kd * (vref - v)
            v_cmd = clamp_norm(v_cmd, speed_cap)

            c.moveByVelocityAsync(float(v_cmd[0]), float(v_cmd[1]), float(v_cmd[2]),
                                  DT,
                                  drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                  yaw_mode=airsim.YawMode(False, 0.0),
                                  vehicle_name=veh_name)

            # stop-on-hit
            if stop_mode != "none" and t >= hit_cooldown:
                margin = hit_margin_ue(c, hit_drone_ue_key, hit_target_ue_key, hit_rv, hit_rt) if stop_mode == "ue" \
                         else hit_margin_airsim(c, hit_drone_veh, hit_target_veh, hit_rv, hit_rt)
                if margin is not None and margin <= 0.0:
                    print(f"[stop] HIT ({stop_mode}) at t={t:.2f}s  margin={margin:.3f} m — stopping mover.")
                    break

            if debug > 0 and (time.time() - t0) >= next_dbg:
                next_dbg += debug
                print(f"t={t:5.2f}  |p-p_ref|={np.linalg.norm(pref-p):6.2f}  v_cmd=({v_cmd[0]:5.2f},{v_cmd[1]:5.2f},{v_cmd[2]:5.2f})")

            time.sleep(DT)
    finally:
        # Put the vehicle in hover to visibly stop, then return control
        try: c.hoverAsync(vehicle_name=veh_name).join()
        except Exception: pass

def move_target_ue(c, ue_key, gen, params, tmax, debug,
                   stop_mode, hit_drone_ue_key, hit_target_ue_key,
                   hit_drone_veh, hit_target_veh, hit_rv, hit_rt, hit_cooldown):
    t0 = time.time(); next_dbg = 0.0
    try:
        base_pose = c.simGetObjectPose(ue_key); q = base_pose.orientation
    except Exception:
        q = airsim.Quaternionr(0,0,0,1)
    while True:
        t = time.time() - t0
        if t > tmax: break

        pref, _ = gen(t, **params)
        pose = airsim.Pose(airsim.Vector3r(float(pref[0]), float(pref[1]), float(pref[2])), q)
        c.simSetObjectPose(ue_key, pose, teleport=True)

        # stop-on-hit
        if stop_mode != "none" and t >= hit_cooldown:
            margin = hit_margin_ue(c, hit_drone_ue_key, hit_target_ue_key, hit_rv, hit_rt) if stop_mode == "ue" \
                     else hit_margin_airsim(c, hit_drone_veh, hit_target_veh, hit_rv, hit_rt)
            if margin is not None and margin <= 0.0:
                print(f"[stop] HIT ({stop_mode}) at t={t:.2f}s  margin={margin:.3f} m — stopping mover.")
                break

        if debug > 0 and (time.time() - t0) >= next_dbg:
            next_dbg += debug
            print(f"t={t:5.2f}  UE set=({pref[0]:6.2f},{pref[1]:6.2f},{pref[2]:6.2f})")

        time.sleep(DT)

# -------------------- main --------------------
def main():
    p = argparse.ArgumentParser(description="Target mover (circle/line/lemniscate) for AirSim vehicle or UE actor")

    # Mode + object selection
    p.add_argument("--mode", choices=["airsim","ue"], default="airsim")
    p.add_argument("--veh", type=str, default="Target", help="AirSim vehicle name (mode=airsim)")
    p.add_argument("--ue-name", type=str, default="Target", help="UE actor name (mode=ue)")

    # Pattern + params
    p.add_argument("--pattern", choices=list(PATTERNS.keys()), default="circle")
    p.add_argument("--radius", type=float, default=15.0)
    p.add_argument("--omega",  type=float, default=0.12)
    p.add_argument("--z",      type=float, default=-15.0)
    p.add_argument("--center-x", type=float, default=0.0)
    p.add_argument("--center-y", type=float, default=0.0)

    # Controller (airsim mode)
    p.add_argument("--speed-cap", type=float, default=6.0)
    p.add_argument("--kp", type=float, default=1.0)
    p.add_argument("--kd", type=float, default=0.3)
    p.add_argument("--place-on-start", action="store_true")

    # Stop-on-hit
    p.add_argument("--stop-on-hit", choices=["ue","airsim","none"], default="ue")
    p.add_argument("--hit-drone-ue",   type=str, default="Drone1")
    p.add_argument("--hit-target-ue",  type=str, default="Target")
    p.add_argument("--hit-drone-veh",  type=str, default="Drone1")
    p.add_argument("--hit-target-veh", type=str, default="Target")
    p.add_argument("--hit-rv", type=float, default=0.8)
    p.add_argument("--hit-rt", type=float, default=0.8)
    p.add_argument("--hit-cooldown", type=float, default=0.0, help="seconds to ignore hit checks at start")

    # Misc
    p.add_argument("--tmax", type=float, default=120.0)
    p.add_argument("--debug", type=float, default=0.5)
    p.add_argument("--dump-scene", type=int, default=0, help="print first N scene object names (UE) and exit")

    args = p.parse_args()

    # Connect
    c = airsim.MultirotorClient(ip="127.0.0.1", port=PORT)
    c.confirmConnection(); print("Connected!")

    # Optional scene dump
    if args.dump_scene:
        names = c.simListSceneObjects(".*") or []
        print(f"[ue] total objects: {len(names)}")
        for s in names[:args.dump_scene]: print(s)
        sys.exit(0)

    # Prepare generators and params
    gen = PATTERNS[args.pattern]
    params = dict(cx=args.center_x, cy=args.center_y, r=args.radius, omega=args.omega, z=args.z)

    # Resolve UE keys if we’ll use UE distance for stop-on-hit or UE mode
    hit_drone_ue_key = hit_target_ue_key = None
    ue_key = None
    if args.stop_on_hit == "ue" or args.mode == "ue":
        hit_drone_ue_key  = resolve_scene_key(c, args.hit_drone_ue)
        hit_target_ue_key = resolve_scene_key(c, args.hit_target_ue)
        if args.mode == "ue":
            ue_key = hit_target_ue_key if hit_target_ue_key and args.hit_target_ue == args.ue_name \
                     else resolve_scene_key(c, args.ue_name)
            if not ue_key:
                print("[error] could not resolve UE actor name to move.")
                sys.exit(1)

    # Run mover
    if args.mode == "airsim":
        # make sure we have control
        try:
            c.enableApiControl(True, args.veh); c.armDisarm(True, args.veh)
        except Exception:
            pass
        move_target_airsim(c, args.veh, gen, params, args.speed_cap, args.kp, args.kd,
                           args.place_on_start, args.tmax, args.debug,
                           args.stop_on_hit, hit_drone_ue_key, hit_target_ue_key,
                           args.hit_drone_veh, args.hit_target_veh, args.hit_rv, args.hit_rt, args.hit_cooldown)
        # tidy up a bit
        try:
            c.hoverAsync(vehicle_name=args.veh).join()
        except Exception:
            pass
    else:
        move_target_ue(c, ue_key, gen, params, args.tmax, args.debug,
                       args.stop_on_hit, hit_drone_ue_key, hit_target_ue_key,
                       args.hit_drone_veh, args.hit_target_veh, args.hit_rv, args.hit_rt, args.hit_cooldown)

    # Hard exit so no lingering async tasks keep the process alive
    sys.exit(0)

if __name__ == "__main__":
    main()

