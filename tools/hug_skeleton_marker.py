#!/usr/bin/env python3
"""Publish HUG 21-point skeleton markers for an offline L25 trajectory."""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

PAIRS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

class HugMarker(Node):
    def __init__(self, path: Path, fps: float):
        super().__init__('hug_skeleton_marker')
        with path.open('rb') as stream:
            records = pickle.load(stream)
        self.points = []
        for record in records:
            xyz = record.get('human_keypoints_retarget_frame', record.get('human_keypoints'))
            self.points.append(np.asarray(xyz, dtype=float) if xyz is not None and np.asarray(xyz).shape == (21,3) else None)
        self.pub = self.create_publisher(Marker, '/hug_skeleton', 10)
        self.frame = 0
        self.timer = self.create_timer(1.0 / fps, self.tick)
        self.get_logger().info(f'Publishing HUG skeleton for {len(self.points)} frames on /hug_skeleton')

    def tick(self):
        if self.frame >= len(self.points):
            self.timer.cancel()
            return
        xyz = self.points[self.frame]
        self.frame += 1
        if xyz is None:
            return
        stamp = self.get_clock().now().to_msg()
        lines = Marker()
        lines.header.frame_id = 'hand_base_link'
        lines.header.stamp = stamp
        lines.ns = 'hug_skeleton'
        lines.id = 0
        lines.type = Marker.LINE_LIST
        lines.action = Marker.ADD
        lines.scale.x = 0.003
        lines.color.r, lines.color.g, lines.color.b, lines.color.a = 1.0, 0.15, 0.05, 0.95
        for a, b in PAIRS:
            for i in (a, b):
                p = Point(x=float(xyz[i,0]), y=float(xyz[i,1]), z=float(xyz[i,2]))
                lines.points.append(p)
        self.pub.publish(lines)
        dots = Marker()
        dots.header = lines.header
        dots.ns = 'hug_keypoints'
        dots.id = 1
        dots.type = Marker.SPHERE_LIST
        dots.action = Marker.ADD
        dots.scale.x = dots.scale.y = dots.scale.z = 0.008
        dots.color.r, dots.color.g, dots.color.b, dots.color.a = 1.0, 0.75, 0.05, 1.0
        dots.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in xyz]
        self.pub.publish(dots)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=30.0)
    args = parser.parse_args()
    rclpy.init()
    node = HugMarker(args.trajectory, args.fps)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
