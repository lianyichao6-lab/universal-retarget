#!/usr/bin/env python3
"""Estimate L25 Vector scales from human canonical vectors and L25 FK geometry."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import yaml
from anydexretarget.hand_representation import load_canonical_grasp_state
from anydexretarget.retarget import Retargeter

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CONFIG=ROOT/'example/config/vector/mediapipe/mediapipe_linkerhand_l25.yaml'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--canonical',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--config-output',type=Path)
    ap.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    args=ap.parse_args()
    state=load_canonical_grasp_state(args.canonical)
    ret=Retargeter.from_yaml(str(args.config),hand_side='right')
    opt=ret.optimizer
    kp=state.keypoints_for_retargeting()
    q_solved, verbose=ret.retarget_verbose(kp,apply_filter=False)
    q_solved=np.asarray(q_solved,dtype=np.float64)
    lower,upper=opt.robot.joint_limits[:,0],opt.robot.joint_limits[:,1]
    q_open=np.clip(np.zeros_like(lower),lower,upper)
    names=list(opt.robot.dof_joint_names)
    points_idx=opt._kv_computed_link_indices
    offsets=opt._kv_computed_link_offsets
    robot_open=opt.robot.compute_points_batch(q_open,points_idx,offsets)
    robot_solved=opt.robot.compute_points_batch(q_solved,points_idx,offsets)
    origin=opt._kv_origin_indices; task=opt._kv_task_indices
    human=np.asarray(kp[opt._origin_kp_indices]-kp[opt._task_kp_indices],dtype=float)
    human=-human
    human_len=np.linalg.norm(human,axis=1)
    current=np.asarray(opt._vector_scalings,dtype=float)
    rows=[]
    candidate=[]
    for i,entry in enumerate(yaml.safe_load(args.config.read_text())['retarget']['key_vectors']):
        ro=robot_open[task[i]]-robot_open[origin[i]]
        rs=robot_solved[task[i]]-robot_solved[origin[i]]
        lo=float(np.linalg.norm(ro)); ls=float(np.linalg.norm(rs)); hl=float(human_len[i])
        co=lo/hl if hl>1e-9 else float('nan')
        cs=ls/hl if hl>1e-9 else float('nan')
        candidate.append(co)
        rows.append({'index':i,'origin':entry['origin'],'task':entry['task'],'origin_kp':entry['origin_kp'],'task_kp':entry['task_kp'],'current_scale':float(current[i]),'human_length_m':hl,'l25_open_length_m':lo,'l25_solved_length_m':ls,'scale_from_open_fk':co,'scale_from_solved_fk':cs})
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps({'canonical':str(args.canonical.resolve()),'config':str(args.config.resolve()),'reference':'q=0 clipped to URDF limits','rows':rows},indent=2)+'\n')
    with args.report.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if args.config_output:
        data=yaml.safe_load(args.config.read_text())
        for entry,scale in zip(data['retarget']['key_vectors'],candidate): entry['scale']=round(float(scale),6)
        args.config_output.parent.mkdir(parents=True,exist_ok=True)
        args.config_output.write_text(yaml.safe_dump(data,sort_keys=False))
    print(f'Wrote scale report: {args.report}')
    if args.config_output: print(f'Wrote experimental config: {args.config_output}')
    for row in rows: print(f"{row['index']:2d} {row['origin']}->{row['task']} current={row['current_scale']:.3f} open_fk={row['scale_from_open_fk']:.3f} solved_fk={row['scale_from_solved_fk']:.3f}")
if __name__=='__main__': main()
