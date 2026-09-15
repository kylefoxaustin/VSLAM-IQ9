#!/usr/bin/env python3
"""Plot the recovered VO camera trajectory against ground truth.

Input: vo_traj.csv (columns: frame,est_x,est_y,gt_x,gt_y), written by the pipeline harness.
Output: vo_trajectory.png (trajectory overlay + per-frame position error).

MEASURED on the iq9 (16-frame synthetic sequence, camera 1.5 deg/frame rotation + translation,
Harris+orient+rBRIEF on the Hexagon cDSP): estimate tracks ground truth to max 0.15 px, 0.01 deg
heading drift. See docs/FINDINGS.md section 10.
"""
import csv, math, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "vo_traj.csv"
fr=[];ex=[];ey=[];gx=[];gy=[]
with open(src) as f:
    for r in csv.DictReader(f):
        fr.append(int(float(r["frame"]))); ex.append(float(r["est_x"])); ey.append(float(r["est_y"]))
        gx.append(float(r["gt_x"])); gy.append(float(r["gt_y"]))
err=[math.hypot(ex[i]-gx[i],ey[i]-gy[i]) for i in range(len(fr))]

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={"width_ratios":[1.5,1]})
ax1.plot(gx,gy,"o-",color="#888",ms=6,lw=2,label="ground truth")
ax1.plot(ex,ey,"x--",color="#d1495b",ms=8,mew=2,lw=1.4,label="VO estimate (Hexagon front-end)")
ax1.set_title("Recovered camera trajectory vs ground truth",fontweight="bold")
ax1.set_xlabel("scene x (px)"); ax1.set_ylabel("scene y (px)")
ax1.legend(); ax1.grid(alpha=0.3); ax1.set_aspect("equal","datalim")
ax2.plot(fr,err,"o-",color="#2e8b8b",lw=2); ax2.axhline(1.0,ls=":",color="#aaa",label="1 px")
ax2.set_title(f"Position error (max {max(err):.2f} px)",fontweight="bold")
ax2.set_xlabel("frame"); ax2.set_ylabel("|est - GT| (px)"); ax2.set_ylim(0,max(1.2,max(err)*1.3))
ax2.legend(); ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("vo_trajectory.png",dpi=130)
print("wrote vo_trajectory.png  end-err=%.3f px  max-err=%.3f px"%(err[-1],max(err)))
