# Branch ownership

`feature/hug-multihand-core` owns portable hand functionality: input adapters,
retargeting configurations, HUG candidate generation, mesh-aware hand/object
filtering, MuJoCo playback, and direct vendor hand validation.

It must not acquire robot-arm kinematics, ROS nodes, controller topics, or
Luban-specific launch and execution code.

`feature/ar5-luban-grasp-integration` is layered on this branch. It owns AR5
planning, ROS/Luban command adapters, collision-scene publication, and full
robot execution. Experimental smolVLA and tactile work remains on its existing
legacy branch and is deliberately outside both branches.
