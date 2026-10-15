"""Baseline launch and rosbag recording plans for BEV + VINS experiments."""

import json
import shlex
from dataclasses import asdict, dataclass


BEV_STATUS_TOPIC = "/bev_obstacle_detection/status"
BEV_OCCUPANCY_TOPIC = "/bev_obstacle_detection/occupancy_grid"
BEV_MARKERS_TOPIC = "/bev_obstacle_detection/obstacle_markers"


@dataclass(frozen=True)
class MethodPlan:
    method: str
    description: str
    launch_commands: tuple
    required_topics: tuple
    output_topics: tuple
    record_topics: tuple


def _unique(items):
    seen = set()
    unique_items = []
    for item in items:
        if item and item not in seen:
            unique_items.append(item)
            seen.add(item)
    return tuple(unique_items)


def _topics(config, keys):
    return tuple(str(config.topics.get(key, "")) for key in keys if config.topics.get(key, ""))


def _launch_arg(name, value):
    return "{}:={}".format(name, shlex.quote(str(value)))


def _bev_launch_command(config, launch_file):
    use_radar = "true" if config.topics.get("radar", "") else "false"
    point_cloud_transform = config.topics.get("point_cloud_transform", "")
    if not point_cloud_transform and "realsense/depth_camera/depth/points" in config.topics.get("lidar", ""):
        point_cloud_transform = "realsense_depth_to_body"
    args = [
        _launch_arg("camera_topic", config.topics.get("camera", "")),
        _launch_arg("lidar_topic", config.topics.get("lidar", "")),
        _launch_arg("radar_topic", config.topics.get("radar", "")),
        _launch_arg("odom_topic", config.topics.get("vins_odom", "")),
        _launch_arg("use_radar", use_radar),
        _launch_arg("point_cloud_transform", point_cloud_transform or "none"),
    ]
    return "roslaunch bev_obstacle_detection {} {}".format(launch_file, " ".join(args))


def method_plan(config, method):
    control_topics = _topics(config, ("control_cmd", "control_pose_enu", "control_vel_flu"))

    if method == "vins_only":
        required = _topics(config, ("camera", "camera_right", "imu"))
        outputs = _topics(config, ("vins_odom", "vins_camera_pose", "gazebo_gt"))
        return MethodPlan(
            method=method,
            description="VINS-Fusion stereo+IMU localization baseline without BEV obstacle logic.",
            launch_commands=("roslaunch vins xtdrone_run_vio.launch",),
            required_topics=required,
            output_topics=outputs,
            record_topics=_unique(required + outputs + control_topics),
        )

    if method == "bev_only":
        required = _topics(config, ("camera", "lidar", "radar"))
        outputs = _topics(config, ("gazebo_gt",)) + (BEV_STATUS_TOPIC, BEV_OCCUPANCY_TOPIC, BEV_MARKERS_TOPIC)
        return MethodPlan(
            method=method,
            description="BEV obstacle baseline using sensor inputs without VINS odometry as a required modality.",
            launch_commands=(_bev_launch_command(config, "bev_only.launch"),),
            required_topics=required,
            output_topics=outputs,
            record_topics=_unique(required + outputs + control_topics),
        )

    if method == "bev_vins_fusion":
        required = _topics(config, ("camera", "camera_right", "lidar", "radar", "imu", "vins_odom"))
        outputs = _topics(config, ("vins_camera_pose", "gazebo_gt")) + (
            BEV_STATUS_TOPIC,
            BEV_OCCUPANCY_TOPIC,
            BEV_MARKERS_TOPIC,
        )
        return MethodPlan(
            method=method,
            description="Fusion baseline that uses BEV sensor inputs plus VINS odometry for temporal alignment and robustness checks.",
            launch_commands=(
                "roslaunch vins xtdrone_run_vio.launch",
                _bev_launch_command(config, "bev_vins_fusion.launch"),
            ),
            required_topics=required,
            output_topics=outputs,
            record_topics=_unique(required + outputs + control_topics),
        )

    raise ValueError("unsupported baseline method: {}".format(method))


def baseline_plan(config):
    return tuple(method_plan(config, method) for method in config.methods)


def plan_to_json(plan):
    return [asdict(item) for item in plan]


def write_plan_json(path, plan):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(plan_to_json(plan), stream, indent=2, sort_keys=True)
        stream.write("\n")


def runbook_markdown(config, plan):
    lines = [
        "# Baseline Runbook",
        "",
        "Experiment: `{}`".format(config.experiment_id),
        "",
        "This runbook keeps VINS-only, BEV-only, and BEV+VINS fusion recordings comparable.",
        "",
        "## Common Setup",
        "",
        "```bash",
        "source /opt/ros/noetic/setup.bash",
        "source {}/devel/setup.bash".format(config.workspace_root),
        "```",
        "",
        "Start the XTDrone/PX4/Gazebo scenario first, then run one baseline method at a time.",
        "",
    ]

    for item in plan:
        lines.extend(
            [
                "## {}".format(item.method),
                "",
                item.description,
                "",
                "Launch:",
                "",
                "```bash",
            ]
        )
        lines.extend(item.launch_commands)
        lines.extend(
            [
                "```",
                "",
                "Required topics:",
                "",
            ]
        )
        lines.extend("- `{}`".format(topic) for topic in item.required_topics)
        lines.extend(["", "Record command:", "", "```bash"])
        lines.append(
            "METHOD={} ENVIRONMENT=clear DURATION=60 ./baseline_record.sh".format(
                shlex.quote(item.method)
            )
        )
        lines.extend(["```", ""])

    return "\n".join(lines).rstrip() + "\n"


def record_script(config, plan):
    methods = " ".join(shlex.quote(item.method) for item in plan)
    environments = " ".join(shlex.quote(item) for item in config.environments)
    datasets = " ".join(shlex.quote(item.name) for item in config.datasets)
    default_bag_root = str(config.datasets[0].root).rstrip("/") + "/rosbags"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "WORKSPACE_ROOT={}".format(shlex.quote(config.workspace_root)),
        "BAG_ROOT=${BAG_ROOT:-%s}" % shlex.quote(default_bag_root),
        "METHOD=${METHOD:-vins_only}",
        "ENVIRONMENT=${ENVIRONMENT:-clear}",
        "DATASET=${DATASET:-%s}" % shlex.quote(config.datasets[0].name),
        "DURATION=${DURATION:-60}",
        "COMPRESSION=${COMPRESSION:-lz4}",
        "",
        "source /opt/ros/noetic/setup.bash",
        "source \"$WORKSPACE_ROOT/devel/setup.bash\"",
        "",
        "KNOWN_METHODS=(%s)" % methods,
        "KNOWN_ENVIRONMENTS=(%s)" % environments,
        "KNOWN_DATASETS=(%s)" % datasets,
        "",
        "contains() {",
        "  local needle=$1",
        "  shift",
        "  for item in \"$@\"; do",
        "    if [[ \"$item\" == \"$needle\" ]]; then",
        "      return 0",
        "    fi",
        "  done",
        "  return 1",
        "}",
        "",
        "contains \"$METHOD\" \"${KNOWN_METHODS[@]}\" || { echo \"unknown METHOD: $METHOD\" >&2; exit 2; }",
        "contains \"$ENVIRONMENT\" \"${KNOWN_ENVIRONMENTS[@]}\" || { echo \"unknown ENVIRONMENT: $ENVIRONMENT\" >&2; exit 2; }",
        "contains \"$DATASET\" \"${KNOWN_DATASETS[@]}\" || { echo \"unknown DATASET: $DATASET\" >&2; exit 2; }",
        "",
        "case \"$METHOD\" in",
    ]

    for item in plan:
        lines.append("  {})".format(shlex.quote(item.method)))
        lines.append("    TOPICS=(%s)" % " ".join(shlex.quote(topic) for topic in item.record_topics))
        lines.append("    ;;")

    lines.extend(
        [
            "esac",
            "",
            "OUT_DIR=\"$BAG_ROOT/$DATASET/$ENVIRONMENT\"",
            "mkdir -p \"$OUT_DIR\"",
            "OUT_BAG=\"$OUT_DIR/${METHOD}.bag\"",
            "",
            "echo \"recording $METHOD / $DATASET / $ENVIRONMENT\"",
            "echo \"output: $OUT_BAG\"",
            "echo \"compression: $COMPRESSION\"",
            "ROSBAG_ARGS=(-O \"$OUT_BAG\" --duration=\"$DURATION\")",
            "case \"$COMPRESSION\" in",
            "  none|\"\")",
            "    ;;",
            "  lz4)",
            "    ROSBAG_ARGS+=(--lz4)",
            "    ;;",
            "  bz2)",
            "    ROSBAG_ARGS+=(-j)",
            "    ;;",
            "  *)",
            "    echo \"unknown COMPRESSION: $COMPRESSION\" >&2",
            "    exit 2",
            "    ;;",
            "esac",
            "rosbag record \"${ROSBAG_ARGS[@]}\" \"${TOPICS[@]}\"",
            "",
        ]
    )
    return "\n".join(lines)


def replay_script(config):
    default_bag_root = str(config.datasets[0].root).rstrip("/") + "/rosbags"
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "WORKSPACE_ROOT={}".format(shlex.quote(config.workspace_root)),
            "BAG_ROOT=${BAG_ROOT:-%s}" % shlex.quote(default_bag_root),
            "METHOD=${METHOD:-vins_only}",
            "ENVIRONMENT=${ENVIRONMENT:-clear}",
            "DATASET=${DATASET:-%s}" % shlex.quote(config.datasets[0].name),
            "",
            "source /opt/ros/noetic/setup.bash",
            "source \"$WORKSPACE_ROOT/devel/setup.bash\"",
            "",
            "BAG=\"$BAG_ROOT/$DATASET/$ENVIRONMENT/${METHOD}.bag\"",
            "test -f \"$BAG\" || { echo \"bag not found: $BAG\" >&2; exit 2; }",
            "rosbag play --clock \"$BAG\"",
            "",
        ]
    )
