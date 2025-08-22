# Intercept Demo (UE5 + AirSim)

<img width="1450" height="816" alt="image" src="https://github.com/user-attachments/assets/8bb2bdb6-0b4b-4809-a071-45b3b0c76cce" />

Short Video below
[![Watch the video](https://img.youtube.com/vi/Cawvy5rIuoU/0.jpg)](https://youtu.be/Cawvy5rIuoU)

This repo contains two Python tools that run against an Unreal (UE5) + AirSim build to demo single‑vehicle **intercept** against a moving **target**. Everything is **headless‑first** and scriptable so it fits both manual runs and CI/RL pipelines.

# Why UE5 and AirSim

Photoreal visuals → better sim-to-real.
AirSim is built on Unreal Engine, so you get physically based materials, advanced lighting/shadows, and a massive Marketplace of high-detail assets (many via photogrammetry). That visual richness is exactly what vision models need for transfer.
<img width="1365" height="773" alt="image" src="https://github.com/user-attachments/assets/4cb84b41-1f18-40cc-b22a-02d1c6149784" />
<img width="1367" height="771" alt="image" src="https://github.com/user-attachments/assets/5708fb29-2009-414c-94c4-ac8d33353e98" />

Built-in ground-truth for ML (RGB, depth, segmentation).
Out of the box, AirSim can stream RGB plus synchronized depth and semantic segmentation frames in real time—perfect for supervised learning and perception debugging. The AirSim paper literally shows the three feeds side-by-side.
<img width="1935" height="1106" alt="image" src="https://github.com/user-attachments/assets/efb26bd0-2f32-49ef-b742-82978de6d303" />


First-class APIs and HITL/SITL paths.
AirSim exposes a clean API for commanding vehicles and reading sensors, and was designed to run at high frequency for real-time hardware-in-the-loop (HITL) with common protocols (e.g., MAVLink). That makes it practical to take algorithms from sim to the real autopilot stack.

Collision and environment fidelity are “free.”
Unreal’s collision pipeline (penetration depth, normals) feeds straight into AirSim’s physics, so impacts and constraints behave sensibly without extra plumbing.

Reproducible data generation at scale.
Because it’s a UE plugin with a stable API, you can run headless for CI/auto-dataset jobs or windowed for debugging, and keep exact run configs with your logged outputs. (That’s how this repo produces CSV + metrics + replay files automatically.)

Reality check: speed vs. realism trade-off.
If you need extreme RL throughput with many agents, a decoupled renderer/physics stack like Flightmare (Unity) can reach ~230 Hz rendering and up to ~200 kHz physics on a laptop. For this challenge, we favor UE photorealism and dense ground-truth labels over raw simulation FPS.

Evidence of transfer.
The AirSim team compared simulated flights to real-world tracks and reported close agreement (e.g., sub-meter Hausdorff distances on some patterns), supporting its use for algorithm development before field tests.

New explainer video with my comments (coming soon)

[![Watch the video](https://img.youtube.com/vi/am2o6xX1B5w/0.jpg)](https://youtu.be/am2o6xX1B5w)


# Reading and sources
https://microsoft.github.io/AirSim/api_docs/html/
https://cosys-lab.github.io/Cosys-AirSim/

- **`intercept.py`** — two‑stage guidance with CSV/JSON logging, optional wind/noise, UE line‑strip traces, and `replay.json` export.
- **`target_mover.py`** — moves the target as an AirSim vehicle (PD velocity controller) or directly teleports a UE actor along simple patterns (circle / line / lemniscate), with optional stop‑on‑hit.

Both scripts talk to AirSim on **port 41451** using the Cosys‑AirSim Python client (`import cosysairsim as airsim`).

# UE5.5 binary
https://drive.google.com/drive/folders/1fizMTQhh68QP4HP47nf80SIxJHJ3IVHN?usp=sharing

<img width="257" height="72" alt="image" src="https://github.com/user-attachments/assets/3f3099a4-74a9-4ecb-b677-88539b332a1a" />

## UE5 run flags (headless vs windowed)

You can run the packaged UE5.5 binary either headless for servers/CI or in a window for debugging.

**Headless (NullRHI):**
```powershell
.\ -NullRHI -Unattended -NoSplash -NoSound -Log
```

**Windowed 1080p (useful for dev):**
```powershell
.\ -Windowed -ResX=1920 -ResY=1080 -Log
```
> The shared Drive folder contains the UE5.5 build and two shortcuts pre‑configured for these modes. Default build opens fullscreen; use the windowed flags above to override.

## Quick start

1) **Launch UE + AirSim** (headless or windowed).  
2) **Move a target** and **fly the interceptor** in one shell line:

**UE actor as target (teleport), stop on UE hit**
```bash
python target_mover.py --mode ue --ue-name Target \
  --pattern circle --radius 20 --omega 0.1 --z -20 \
  --stop-on-hit ue & \
python intercept.py --guidance-source ue --ue-target-name Target \
  --ue-drone-name Drone1 --write-metrics-json
```

**AirSim vehicle as target (truth in dynamics space), stop on AirSim hit**
```bash
python target_mover.py --mode airsim --veh Target \
  --pattern lemniscate --omega 0.15 --z -12 --stop-on-hit airsim & \
python intercept.py --guidance-source airsim \
  --as-target-name Target --as-drone-name Drone1 --write-metrics-json
```

> The interceptor always uses **AirSim truth** for its own state; the target pose comes from **UE** (by actor name) or **AirSim** depending on `--guidance-source`.

---

## What each script does

### `intercept.py` — Two‑stage guidance + logging

**Modes**
- **Pursuit (far):** lead‑pursuit PD with capped look‑ahead `tau_max`.
- **Capture (near or overshoot):** velocity matching  
  \\( \mathbf{v}_{cmd} = \mathbf{v}_T + K_p(\mathbf{p}_T-\mathbf{p}_S) - K_d(\mathbf{v}_S-\mathbf{v}_T) \\).

Automatically switches based on `--capture-dist` and opening rate `rdot` (enter capture if close **or** opening while near). You can force capture everywhere with `--always-capture`.

**Pose sources**
- `--guidance-source ue` — target pose from UE actor (velocity estimated online with EMA, see `--vt-ema`).
- `--guidance-source airsim` — target pose/velocity from an AirSim vehicle.

**Hit / end‑of‑run**
- Declares **HIT** when **UE distance ≤ rv+rt** (spherical overlap).  
- After a HIT: optional pause (`--hit-pause`), then **`c.reset()`** the world.  
- Each run creates a folder `logs/<prefix>_YYYYmmdd-HHMMSS/`
  
traj.csv (created in logs/)

<img width="1644" height="283" alt="image" src="https://github.com/user-attachments/assets/025fe46d-adca-46e7-9c9f-17d46d112a21" />

```
t - Simulation time in seconds.
S_as_x, S_as_y, S_as_z - drone position in UE.
S_as_vx, S_as_vy, S_as_vz - velocity vector from AirSim.

S_meas_x, S_meas_y, S_meas_z - guidance position.
S_meas_vx, S_meas_vy, S_meas_vz - measured velocity.
fold in noise models, filters, etc.

T_x, T_y, T_z - target position.
T_vx, T_vy, T_vz - target velocity.

dist_ue - Distance between interceptor and target in UE world space (‖pS − pT‖).
This is what the hit detector uses (compared to rv+rt).

mode -  PURSUIT - far away, proportional pursuit (PD law with capped tau).
        CAPTURE - close in, velocity-matching to damp overshoot.

rdot - Closing/opening rate
      Negative - closing
      Positive → diverging
      tuning tau_max and capture_dist
```

created metrics.json (created in logs/)
    
```
hit — true if a collision was declared (distance ≤ rv+rt).
t_end — simulation time when the run ended. If hit=true, this is the time-to-intercept.
min_dist — closest approach distance between interceptor and target (meters).
guidance_source — pose source for the target: "ue" (UE actor) or "airsim" (AirSim truth).
mode_final — last guidance mode reached (PURSUIT or CAPTURE).
capture_dist — threshold (m) where controller switches from PURSUIT to CAPTURE.
rv, rt — collision radii (m) for interceptor and target.
speed_cap_xy, speed_cap_z — horizontal and vertical velocity caps (m/s).
kp, kd — PD gains for PURSUIT mode.
rel_kp, rel_kd — gains for CAPTURE mode (velocity matching).
tau_max — max look-ahead time for pursuit guidance.
wind — constant wind bias [wx, wy, wz] applied to measured velocities.
noise_pos, noise_vel — Gaussian noise levels (σ) added to measured position (m) and velocity (m/s).
run_dir — output folder containing logs for this run.
csv — path to the full trajectory log (traj.csv).
```

replay.json` (created in logs/)
 
 ```
t — list of simulation times (s).
S — list of interceptor positions at each time (AirSim truth).
T — list of target positions at each time (UE actor or AirSim vehicle depending on --guidance-source).
```

  - optional PNG plots.
<p float="left">
  <img src="https://github.com/user-attachments/assets/e0804d7b-1bd5-40a2-b776-cf24d87eff0a" height="150" width="250" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/e172f9d0-a93d-4610-bc04-353992cb9030"  height="150" width="250" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/46eb4e3a-9013-4430-93ef-c79558f1a06e" height="150" width="250" />
</p>


**UE traces**
- Draws persistent line‑strips (drone in blue, target in red). Tune with `--trace-every` and `--trace-duration`. Traces are flushed between runs.

**Wind & measurement noise (optional)**
- Add constant wind bias and Gaussian noise on measured pos/vel only (guidance still flies on the **measured** states).
- `--wind "wx,wy,wz"` (NED), `--noise-pos`, `--noise-vel`, `--wind-applies-to meas|none`.

CLI (key flags & defaults)
```
--guidance-source ue|airsim        (default: ue)
--ue-target-name Target            --ue-drone-name Drone1
--as-target-name Target            --as-drone-name Drone1
--speed-cap-xy 14.0                --speed-cap-z 6.0

# Pursuit PD (far)
--kp 1.2 --kd 0.35 --tau-max 2.0

# Capture (near / overshoot)
--capture-dist 8.0 --rel-kp 1.2 --rel-kd 0.8 --always-capture

# UE velocity estimate
--vt-ema 0.5

# Collision radii (UE distance)
--rv 0.8 --rt 0.8 --hit-pause 2.0

# Traces
--trace-every 5 --trace-duration 0.0

# Logging
--log-dir logs --csv-prefix run --no-csv (off) --write-metrics-json (on)

# Wind/Noise
--wind "0,0,0" --noise-pos 0.0 --noise-vel 0.0 --wind-applies-to meas

# Misc
--tmax 60.0 --debug 0.3
```

---

target_mover.py — Move the target (AirSim or UE actor)

**Patterns**
- `circle` (radius/omega/z, optional center),
- `line` (ping‑pong between 2 points, fixed T),
- `lemniscate` (figure‑8).

**Modes**
- `--mode airsim` — PD velocity controller to track the pattern (caps at `--speed-cap`).
- `--mode ue` — teleports a UE actor along the pattern (`simSetObjectPose`).

**Stop‑on‑hit (optional)**
- `--stop-on-hit ue|airsim|none` — compute spherical hit using either **UE** actor positions or **AirSim** vehicle states.
- On hit (after `--hit-cooldown` seconds), the mover stops/hover and exits with code 0.

**Name resolution helpers**
- UE actor names are resolved by **exact name first**, then substring match. Use `--dump-scene N` to list candidates.

#### CLI (key flags & defaults)
```
--mode airsim|ue                   (default: airsim)
--veh Target                       # AirSim vehicle name
--ue-name Target                   # UE actor/substring

# Pattern selection + params
--pattern circle|line|lemniscate   --radius 15 --omega 0.12 --z -15
--center-x 0 --center-y 0

# AirSim PD controller
--speed-cap 6.0 --kp 1.0 --kd 0.3 --place-on-start

# Stop on hit (spherical collision)
--stop-on-hit ue|airsim|none       (default: ue)
--hit-drone-ue Drone1  --hit-target-ue Target
--hit-drone-veh Drone1  --hit-target-veh Target
--hit-rv 0.8 --hit-rt 0.8 --hit-cooldown 0.0

# Misc
--tmax 120 --debug 0.5 --dump-scene 0
```

---

## Outputs to check

- **`min_dist`** — best closest approach (meters) across the run (from `metrics.json`).
- **`t_end`** — wall‑clock time when the run ended. If `hit=true`, this is your **time‑to‑intercept**.
- **`mode` timeline** — `traj.csv` logs `PURSUIT` → `CAPTURE` transitions.
- **`rdot`** in CSV — opening/closing rate (positive = diverging). Great for tuning `tau_max` and `capture_dist`.
- **UE line‑strips** — quick visual of geometry and where capture began.

CSV columns (ordered): time, interceptor (AirSim truth + measured), target (pose/vel), `dist_ue`, `mode`, `rdot`.

---

Troubleshooting

- **UE actor names don’t resolve**  
  Run `--dump-scene 250` to print the first 250 names; or use a shorter substring (exact match is preferred automatically).
- **Drones touch in viewport but no hit**  
  Increase `--rv/--rt` slightly (meshes may not be 1:1 scale). Hit is computed in **UE** space when `--stop-on-hit ue`, otherwise by **AirSim** positions for `airsim`.
- **Controller oscillations**  
  Reduce `--kp`, increase `--kd`, and lower `--speed-cap-xy` / `--speed-cap` (mover). In capture, try `--rel-kd 1.0+` for faster damping.
- **Old traces hanging around**  
  `intercept.py` flushes persistent lines at start and after HIT. If you see leftovers, restart the UE session.

---

## Design choices

- **Two‑stage law** — pursuit first for big geometry error; velocity‑matching capture to avoid overshoot.
- **Dual pose sources** — UE actors for fast prototyping/visuals; fallback to AirSim truth seamlessly.
- **Headless‑first** — CLI‑only operation, all outputs to CSV/JSON/PNG to fit CI and future RL sweeps.

---

## Dev notes

- Python 3.9+ recommended.
- Requires **Cosys-AirSim** Python client (`cosysairsim` import). Ensure UE build runs the compatible AirSim server on **127.0.0.1:41451**.
- Both scripts catch most API errors and will try to `hoverAsync`/`landAsync` and disarm on exit.
