# a2_orchestrator — Autonomous Mission Package

This package runs the **autonomous survey mission**: stand up the robot, explore with TARE, optionally investigate detected objects with FAR, save a DLIO map, and navigate home.

The orchestrator assumes the **mega stack** (TARE + FAR + terrain + `waypoint_mux`, and optionally detection) is already running. It controls locomotion modes and switches planners via topics — no subprocess stack swapping.

---

## Quick Start

```bash
# Terminal 1 — sim
export A2_MODE=sim
a2 sim

# Terminal 2 — DLIO + mega stack (TARE + FAR + mux)
a2 dlio
ros2 launch a2_ros mega.launch.py use_sim_time:=true rviz:=false

# Terminal 3 — mission orchestrator
ros2 run a2_orchestrator mission_orchestrator --ros-args \
    -p save_dir:=./runs/runX \
    -p use_sim_time:=true

# Monitor progress
ros2 topic echo /mission/status
```

### With object detection and investigate

```bash
# Terminal 2 — mega with YOLO + detection_processor
ros2 launch a2_ros mega.launch.py \
  use_sim_time:=true \
  rviz:=false \
  enable_detection:=true \
  sim_detection:=true \
  detection_csv:=/tmp/run1/detections.csv

# Terminal 3 — mission (orchestrator enables detection on EXPLORING entry)
ros2 run a2_orchestrator mission_orchestrator --ros-args \
    -p save_dir:=./runs/runX \
    -p use_sim_time:=true
```

Real robot: start `nuc.launch.py` + pc2 bridge, DLIO, and mega stack (`enable_detection:=true`, `sim_detection:=false`), then run the orchestrator.

---

## Package Layout

```
a2_orchestrator/
├── README.md
├── config/
│   └── mission_defaults.yaml
├── launch/
│   └── mission.launch.py              ← DLIO + orchestrator (optional entry)
└── a2_orchestrator/
    ├── mission_state.py               ← state enum
    ├── mission_orchestrator.py        ← main state machine node
    ├── detection_processor.py         ← YOLO → investigate/resume points
    └── waypoint_mux.py                ← TARE/FAR waypoint multiplexer
```

**Prerequisite stack** (in `a2_ros`):

- `mega.launch.py` — terrain, local planner, TARE, FAR, `waypoint_mux`, optional detection

---

## How It Works

### High-Level Flow

```
Prerequisites (sim + mega + DLIO)  →  mission_orchestrator
                                           │
                                           └─ state machine:
                                                stand → unlock → walk
                                                → select TARE + start exploration
                                                → explore (and investigate objects)
                                                → save map (SavePCD)
                                                → select FAR + goal (0,0,0)
                                                → arrive → sit down → done
```

### Object Detection and Investigate

When `enable_detection:=true` in mega launch, two nodes run:

1. **`object_detection_node`** (same as `a2 detect`) — YOLO on `/detection_info`
2. **`detection_processor`** — tracks objects, publishes `/investigate_point`

Detection processing is **disabled at startup**. The orchestrator is the sole publisher of `/detection/enable`:

| Orchestrator event | `/detection/enable` |
|--------------------|---------------------|
| Node init | `false` |
| Enter `EXPLORING` or `INVESTIGATING` | `true` |
| Enter `SAVE_MAP` | `false` (after `/detection/save`) |

**`/detection/save`** (`std_msgs/Bool`): orchestrator publishes `true` when entering `SAVE_MAP` (exploration done, before map save and nav home). `detection_processor` writes `detections.csv` immediately and does **not** wait for node shutdown.

While disabled, `detection_processor` ignores all detections (no tracking, no investigate points). This prevents a object visible during stand/walk from triggering investigation.

**`/investigate_point` signals** (from `detection_processor` → orchestrator):

| Message | Orchestrator action |
|---------|---------------------|
| Origin `(0, 0, 0)` | Select TARE, resume `EXPLORING` |
| Non-zero map point | Select FAR, publish `/goal_point`, enter `INVESTIGATING` |

Manual enable (testing):

```bash
ros2 topic pub --once /detection/enable std_msgs/msg/Bool "{data: true}"
```

### Motion During Exploration

```
tare_planner  →  /tare/way_point  →  waypoint_mux  →  /way_point  →  localPlanner  →  ...
```

Orchestrator publishes `/planner/select` = `tare` and `/start_exploration` = `true`.

### Motion During Investigate or Return Home

```
mission_orchestrator  →  /planner/select far
                     →  /goal_point (object or home)
far_planner  →  /far/way_point  →  waypoint_mux  →  /way_point  →  localPlanner  →  ...
```

On tare→far switch, `waypoint_mux` publishes a stop goal at the current pose before FAR takes over.

---

## State Machine

| State | What happens |
|-------|----------------|
| `CHECK_PREREQS` | Wait for `/a2/set_mode` and odometry |
| `STAND` → `WAIT_STAND` → `UNLOCK` → `WALK` | Locomotion mode sequence (2→3→4) |
| `RECORD_HOME` | Write `{save_dir}/origin.txt` (actual start pose) |
| `START_EXPLORE` | `/planner/select` = `tare`, `/start_exploration` = `true` |
| `EXPLORING` | Wait for `/exploration_finish` or timeout; `/detection/enable` = `true` |
| `INVESTIGATING` | FAR navigates to detected object; resume via origin on `/investigate_point` |
| `SAVE_MAP` | `/detection/save` then `/detection/enable` false; async SavePCD → `clean_map.pcd` |
| `NAV_HOME` | `/planner/select` = `far`, `/goal_point` at home goal (default 0,0,0) |
| `SIT_DOWN` | Modes 3 → 1 (`BALANCE_STAND` then `STAND_DOWN`) |
| `DONE` | Mission complete |

Status is published on `/mission/status` as `STATE:detail`.

---

## Key Modules

### `mission_orchestrator.py`

Main ROS node. Publishes:

| Topic | Message | Purpose |
|-------|---------|---------|
| `/planner/select` | `std_msgs/String` | `tare` or `far` |
| `/start_exploration` | `std_msgs/Bool` | Start TARE when `kAutoStart: false` |
| `/goal_point` | `PointStamped` | FAR navigation goal |
| `/mission/status` | `std_msgs/String` | State machine status |
| `/detection/enable` | `std_msgs/Bool` | Enable/disable detection processing |
| `/detection/save` | `std_msgs/Bool` | Write detections CSV (on `SAVE_MAP` entry) |

Subscribes:

| Topic | Message | Purpose |
|-------|---------|---------|
| `/investigate_point` | `PointStamped` | Investigate object or resume exploration |

**Do not** `declare_parameter('use_sim_time')` — launch pre-declares it.

### `detection_processor.py`

Subscribes to `/detection_info`, `/detection/enable`, and `/detection/save`. When enabled, tracks objects and publishes `/investigate_point`. Writes `detections.csv` when the orchestrator publishes on `/detection/save` (typically at `SAVE_MAP`, before nav home).

### `waypoint_mux.py`

Runs inside `mega.launch.py`. Forwards TARE or FAR waypoints to `/way_point`. Listens on `/planner/select`.

Manual switch:

```bash
ros2 topic pub --once /planner/select std_msgs/msg/String "{data: 'far'}"
ros2 topic pub --once /planner/select std_msgs/msg/String "{data: 'tare'}"
```

---

## Parameters

Defaults in `config/mission_defaults.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `save_dir` | `/tmp/a2_mission` | Output directory |
| `exploration_timeout_sec` | `600` | Max explore+investigate time (seconds) |
| `skip_home` | `false` | Skip return navigation after map save |
| `home_arrival_threshold_m` | `0.5` | Distance to home goal considered "arrived" |
| `home_goal_x/y/z` | `0.0` | FAR return-home goal in `map` frame |
| `planner_select_topic` | `/planner/select` | Mux control topic |
| `investigate_point_topic` | `/investigate_point` | Investigate/resume signal |
| `detection_enable_topic` | `/detection/enable` | Enable detection processing |
| `detection_save_topic` | `/detection/save` | Save detections CSV |
| `nav_home_timeout_sec` | `600` | Max time for return navigation |
| `stand_wait_sec` | `4.0` | Pause after stand before unlock |
| `dlio_save_pcd_service` | `/save_pcd` | Map save service |

Mega launch detection args:

| Argument | Default | Description |
|----------|---------|-------------|
| `enable_detection` | `false` | Start YOLO + detection_processor |
| `sim_detection` | `false` | Use sim object_detection launch |
| `object_detection_classes` | `[25]` | COCO class IDs (25=umbrella) |
| `detection_csv` | `/tmp/a2_mission/detections.csv` | CSV output path |

Override via launch:

```bash
a2 mission save_dir:=/tmp/run1 exploration_timeout_sec:=300 skip_home:=true
```

---

## Outputs

| File | Description |
|------|-------------|
| `origin.txt` | Actual start pose `x y z` |
| `clean_map.pcd` | DLIO voxel map |
| `detections.csv` | Tracked object detections (when detection enabled) |

---

## Troubleshooting

### Robot doesn't move during exploration

1. Ensure mega stack is running and mux is on `tare`: check `/mission/status`
2. TARE needs `kAutoStart: false` if you rely on `/start_exploration` only
3. Check `/tare/way_point` and `/way_point` have traffic

### Exploration never finishes

- TARE publishes `/exploration_finish` when done
- Or mission stops at `exploration_timeout_sec`

### Investigate never triggers

1. Launch mega with `enable_detection:=true`
2. Confirm `/detection/enable` is `true` during `EXPLORING` (orchestrator publishes on state entry)
3. Check `/detection_info` and `/investigate_point` for traffic

### Nav home doesn't start

- FAR visibility graph must be initialized before goals are accepted
- Check `/goal_point` and `/far/way_point` after `SAVE_MAP` state

---

## Modifying the Mission

### Add a new state

1. Add enum value in `mission_state.py`
2. Add `_tick_your_state()` in `mission_orchestrator.py`
3. Register it in the `_tick()` handlers dict
4. Transition from adjacent states

### Change home goal

Set `home_goal_x`, `home_goal_y`, `home_goal_z` in `mission_defaults.yaml` or via launch args.

### Change exploration stop condition

Edit `_tick_exploring()` — currently stops on `/exploration_finish` or timeout.

---

## Build

```bash
colcon build --packages-select a2_orchestrator --symlink-install
source install/setup.bash
```
