# ShadowHand Description Package

This package provides the URDF model, MuJoCo (MJCF) models for the Shadow Dexterous Hand. It also includes configuration files for ROS 2 visualization using RViz.

## Shadow Hand Specifications

The Shadow Dexterous Hand is a highly articulated robotic hand with **20 degrees of freedom (DOF)** across 5 fingers:

- **Thumb (TH)**: 5 joints (THJ5, THJ4, THJ3, THJ2, THJ1)
- **First Finger / Index (FF)**: 4 joints (FFJ4, FFJ3, FFJ2, FFJ1)
- **Middle Finger (MF)**: 4 joints (MFJ4, MFJ3, MFJ2, MFJ1)
- **Ring Finger (RF)**: 4 joints (RFJ4, RFJ3, RFJ2, RFJ1)
- **Little Finger (LF)**: 5 joints (LFJ5, LFJ4, LFJ3, LFJ2, LFJ1)

### Joint Naming Convention

| Finger | J5 (metacarpal) | J4 (abduction) | J3 (MCP flex) | J2 (PIP flex) | J1 (DIP flex) |
|--------|-----------------|----------------|---------------|---------------|---------------|
| Thumb  | THJ5 (rotation) | THJ4 (flex)    | THJ3 (hub)    | THJ2 (middle) | THJ1 (distal) |
| Index  | -               | FFJ4           | FFJ3          | FFJ2          | FFJ1          |
| Middle | -               | MFJ4           | MFJ3          | MFJ2          | MFJ1          |
| Ring   | -               | RFJ4           | RFJ3          | RFJ2          | RFJ1          |
| Little | LFJ5            | LFJ4           | LFJ3          | LFJ2          | LFJ1          |

## Project Structure

- **`urdf/`**: Contains the Unified Robot Description Format files for the Left and Right hands.
    - `left.urdf` / `right.urdf`: Standard URDFs using primitive shapes.
    - `left-ros.urdf` / `right-ros.urdf`: ROS-specific URDFs. Used by RViz and Launch files.
- **`mjcf/`**: Contains MuJoCo XML model files (`left.xml`, `right.xml`) for simulation.
- **`meshes/`**: Directory for mesh files (currently using primitive shapes).
- **`launch/`**: Python launch scripts to visualize the model in RViz.
- **`rviz/`**: Default RViz configuration files.

## 1. MuJoCo Usage

If you only want to view the model in MuJoCo, you don't need to build the ROS package. Just ensure you have the `mujoco` python package installed.

```bash
pip install mujoco
```

### View Right Hand

```bash
python -m mujoco.viewer --mjcf=mjcf/right.xml
```

### View Left Hand

```bash
python -m mujoco.viewer --mjcf=mjcf/left.xml
```

## 2. ROS 2 and RViz Usage

If you want to use this robot in ROS 2 (Humble/Rolling) with RViz visualization, follow these steps.

### 2.1 Copy to Workspace

Copy this package to the `src` directory of your ROS 2 workspace (e.g., `~/ros2_ws/src`):

```bash
cp -r shadowhand_description ~/ros2_ws/src/
```

### 2.2 Install Dependencies

Install required ROS 2 dependencies:

```bash
rosdep install --from-paths src --ignore-src -r -y
```

### 2.3 Build the Package and Source the Environment

Compile the package using colcon:

```bash
colcon build --packages-select shadowhand_description
source install/setup.bash
```

### 2.4 RViz Visualization

These commands will launch robot_state_publisher, joint_state_publisher_gui, and RViz.

**Visualize Right Hand:**

```bash
ros2 launch shadowhand_description display.right.py
```

**Visualize Left Hand:**

```bash
ros2 launch shadowhand_description display.left.py
```

## Joint Limits (in radians)

| Joint | Lower | Upper | Description |
|-------|-------|-------|-------------|
| THJ5  | -1.047 | 1.047 | Thumb rotation |
| THJ4  | 0 | 1.222 | Thumb flexion |
| THJ3  | -0.209 | 0.209 | Thumb hub |
| THJ2  | -0.524 | 0.524 | Thumb middle |
| THJ1  | -0.262 | 1.571 | Thumb distal |
| xFJ4  | -0.349 | 0.349 | Finger abduction |
| xFJ3  | 0 | 1.571 | MCP flexion |
| xFJ2  | 0 | 1.571 | PIP flexion |
| xFJ1  | 0 | 1.571 | DIP flexion |
| LFJ5  | 0 | 0.698 | Little metacarpal |

## Notes

- This model uses primitive geometric shapes (capsules, spheres, boxes) instead of mesh files for simplicity and compatibility.
- The model is based on the Shadow Dexterous Hand specifications but may not be an exact replica.
- For production use with actual Shadow Hand hardware, please refer to the official Shadow Robot Company documentation.
