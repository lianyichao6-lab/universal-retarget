#!/usr/bin/env python3
"""Offline DexPilot global scaling sweep for one L25 canonical grasp."""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import mujoco,numpy as np
from anydexretarget.dex_backend import DexRetargetBackend
from anydexretarget.hand_representation import load_canonical_grasp_state
ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'assets/linkerhand_l25/linkerhand_l25_right_mujoco.xml'
TIP_OFFSETS={"thumb_distal": np.asarray([-0.008849, -0.000018, 0.030758]), "index_distal": np.asarray([-0.015799, -0.000013, 0.022931])}
def names(m): return [mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)]
def main():
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--canonical',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--scales',type=float,nargs='+',default=[.8,1.,1.2,1.4]); args=ap.parse_args()
 s=load_canonical_grasp_state(args.canonical); kp=s.keypoints_for_retargeting(); m=mujoco.MjModel.from_xml_path(str(MODEL)); mnames=names(m); lo,hi=m.jnt_range[:,0],m.jnt_range[:,1]; rows=[]
 for scale in args.scales:
  b=DexRetargetBackend('dexpilot',hand_side='right',scaling_factor=scale); t=time.perf_counter(); q,v=b.retarget(kp); ms=(time.perf_counter()-t)*1000; by={n.lower():i for i,n in enumerate(b.joint_names)}; mapped=np.asarray([q[by[n.lower()]] for n in mnames],float); viol=int(np.count_nonzero((mapped<lo)|(mapped>hi))); clipped=np.clip(mapped,lo+1e-6,hi-1e-6); margin=np.minimum((clipped-lo)/(hi-lo),(hi-clipped)/(hi-lo)); d=mujoco.MjData(m); d.qpos[:]=clipped; mujoco.mj_forward(m,d); tb=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'thumb_distal'); ib=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'index_distal'); thumb_tip=d.xpos[tb]+d.xmat[tb].reshape(3,3)@TIP_OFFSETS["thumb_distal"]; index_tip=d.xpos[ib]+d.xmat[ib].reshape(3,3)@TIP_OFFSETS["index_distal"]; pinch=float(np.linalg.norm(thumb_tip-index_tip)); cost=float(getattr(getattr(b.optimizer,'opt',None),'last_optimum_value',lambda:float('nan'))()); rows.append({'robot':'l25','optimizer':'dexpilot','scaling_factor':scale,'dof':m.njnt,'solve_ms':ms,'limit_violations':viol,'saturated_joints_5pct':int(np.count_nonzero(margin<=.05)),'min_normalized_margin':float(margin.min()),'thumb_index_distance_m':pinch,'solver_cost_same_optimizer_only':cost})
 args.output.mkdir(parents=True,exist_ok=True); (args.output/'summary.json').write_text(json.dumps({'canonical':str(args.canonical.resolve()),'offline_only':True,'results':rows},indent=2)+'\n');
 with (args.output/'summary.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 print(f'Wrote {len(rows)} DexPilot scale rows to {args.output}')
 for r in rows: print(f"  scale={r['scaling_factor']:.2f} solve={r['solve_ms']:.2f}ms limit={r['limit_violations']} sat={r['saturated_joints_5pct']} pinch={r['thumb_index_distance_m']:.4f}m")
if __name__=='__main__': main()
