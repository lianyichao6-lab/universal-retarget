#!/usr/bin/env python3
"""Convert an offline L25 qpos trajectory to LinkerHand 0-255 commands.

Dry-run only: no ROS2, SDK, CAN or serial connection is opened.
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from anydexretarget.hardware_adapter import L25HardwareAdapter, L25_QPOS_JOINTS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args=p.parse_args()
    with args.trajectory.open("rb") as f: records=pickle.load(f)
    if not isinstance(records,list) or not records: raise ValueError("trajectory must be a non-empty list")
    adapter=L25HardwareAdapter(); commands=[]; qposes=[]
    for i, record in enumerate(records):
        if not isinstance(record,dict) or "target" not in record: raise ValueError(f"frame {i} has no target")
        q=np.asarray(record["target"],dtype=np.float64)
        if q.shape != (21,): raise ValueError(f"frame {i} target must have shape (21,), got {q.shape}")
        cmd=adapter.qpos_to_command(q,L25_QPOS_JOINTS)
        qposes.append(q); commands.append(cmd.values)
    qposes=np.asarray(qposes); commands=np.asarray(commands,dtype=np.int64)
    if np.any(commands<0) or np.any(commands>255): raise ValueError("command range violation")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    payload={"trajectory":str(args.trajectory),"hardware":"linkerhand_l25","dry_run":True,"frames":int(len(commands)),"qpos_joint_names":list(L25_QPOS_JOINTS),"sdk_joint_names":[f"sdk_{i}" for i in range(25)],"commands_0_255":commands.tolist(),"command_min":int(commands.min()),"command_max":int(commands.max())}
    args.output.write_text(json.dumps(payload,indent=2)+"\n")
    print(f"L25 dry-run mapping complete: {len(commands)} frames, command shape={commands.shape}, range=[{commands.min()}, {commands.max()}]")
    print(f"output: {args.output}")

if __name__ == "__main__": main()
