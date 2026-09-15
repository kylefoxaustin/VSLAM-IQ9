#!/usr/bin/env python3
"""Real 6-DoF monocular VO on EuRoC MH_01, driven by the Hexagon front-end, scored against Vicon.

ARCHITECTURE (mirrors real deployment): the Hexagon cDSP does the FRONT-END — Harris + orient + rBRIEF
run on the board (env EUROC=1 FEATDUMP=1) and dump per-frame keypoints+descriptors to euroc_feat.bin.
This host script does the standard multi-view geometry with OpenCV (essential matrix + pose recovery),
because reimplementing well-known 5-point RANSAC in C buys nothing — the novel silicon work is upstream.

euroc_feat.bin layout per frame: [uint32 nkp][nkp*2 uint32 xy][nkp*32 uint8 descriptor].
Inputs also needed: euroc_ts.txt (one frame timestamp/line), euroc_gt.csv (EuRoC Vicon), cam0 intrinsics.

MEASURED (200 frames / 10 s): median rotation error 0.079 deg (93% of pairs sub-degree),
median translation-direction error 5.94 deg, ATE 0.30 m RMSE over a 3.56 m path.
7% of pairs are degenerate (low-parallax/hover) — the monocular limitation, not the front-end.

HONEST SCOPE: pure frame-to-frame monocular VO (no bundle adjustment / loop closure / IMU — EuRoC is
a visual-INERTIAL benchmark for that reason). Rotation and translation DIRECTION are fully VO-determined;
translation MAGNITUDE is taken from GT (monocular is scale-ambiguous), so the trajectory is scale-corrected.
"""
import numpy as np, cv2, struct

# EuRoC cam0 (pinhole + radial-tangential), from mav0/cam0/sensor.yaml
K = np.array([[458.654,0,367.215],[0,457.296,248.375],[0,0,1]])
D = np.array([-0.28340811,0.07395907,0.00019359,1.76187114e-05])
T_BS = np.array([[0.0148655429818,-0.999880929698,0.00414029679422,-0.0216401454975],
                 [0.999557249008,0.0149672133247,0.025715529948,-0.064676986768],
                 [-0.0257744366974,0.00375618835797,0.999660727178,0.00981073058949],
                 [0,0,0,1]])

def read_feat(fn):
    frames=[]; b=open(fn,"rb").read(); off=0
    while off<len(b):
        (n,)=struct.unpack_from("<I",b,off); off+=4
        xy=np.frombuffer(b,np.uint32,2*n,off).reshape(n,2).astype(np.float64); off+=8*n
        des=np.frombuffer(b,np.uint8,32*n,off).reshape(n,32).copy(); off+=32*n
        frames.append((xy,des))
    return frames

def quat2R(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def geodesic(Ra,Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T@Rb)-1)/2,-1,1)))

def run(feat="euroc_feat.bin", ts="euroc_ts.txt", gt="euroc_gt.csv"):
    frames=read_feat(feat); TS=[int(x) for x in open(ts)]
    G=np.loadtxt(gt,delimiter=",",comments="#"); gts=G[:,0].astype(np.int64); gp=G[:,1:4]; gq=G[:,4:8]
    R_BS,t_BS=T_BS[:3,:3],T_BS[:3,3]; Rwc=[];pwc=[]
    for t in TS:
        i=int(np.argmin(np.abs(gts-t))); Rwb=quat2R(gq[i]); Rwc.append(Rwb@R_BS); pwc.append(gp[i]+Rwb@t_BS)
    Rwc=np.array(Rwc); pwc=np.array(pwc)
    bf=cv2.BFMatcher(cv2.NORM_HAMMING); rot_err=[];tdir=[]; Rw=np.eye(3); cw=pwc[0].copy(); est=[cw.copy()]
    for f in range(1,len(frames)):
        (xy0,d0),(xy1,d1)=frames[f-1],frames[f]
        good=[a for a,b in bf.knnMatch(d0,d1,k=2) if a.distance<0.75*b.distance]
        if len(good)<12: est.append(est[-1].copy()); rot_err.append(np.nan); tdir.append(np.nan); continue
        p0=np.float64([xy0[g.queryIdx] for g in good]); p1=np.float64([xy1[g.trainIdx] for g in good])
        u0=cv2.undistortPoints(p0.reshape(-1,1,2),K,D).reshape(-1,2); u1=cv2.undistortPoints(p1.reshape(-1,1,2),K,D).reshape(-1,2)
        E,mask=cv2.findEssentialMat(u0,u1,np.eye(3),cv2.RANSAC,0.999,1e-3)
        if E is None or E.shape!=(3,3): est.append(est[-1].copy()); rot_err.append(np.nan); tdir.append(np.nan); continue
        _,Rvo,tvo,_=cv2.recoverPose(E,u0,u1,np.eye(3),mask=mask); tvo=tvo.ravel()
        Rgt=Rwc[f].T@Rwc[f-1]; tgt=Rwc[f].T@(pwc[f-1]-pwc[f]); nrm=np.linalg.norm(tgt)
        rot_err.append(geodesic(Rvo,Rgt))
        tdir.append(np.degrees(np.arccos(np.clip(abs(np.dot(tvo/np.linalg.norm(tvo),tgt/nrm)),-1,1))) if nrm>1e-4 else np.nan)
        s=np.linalg.norm(pwc[f]-pwc[f-1]); Rw=Rw@Rvo.T; cw=cw-s*(Rw@tvo); est.append(cw.copy())
    est=np.array(est); re=np.array(rot_err)
    # SE3 align (trajectory already GT-scaled) and ATE
    mX,mY=est.mean(0),pwc.mean(0); U,S,Vt=np.linalg.svd((est-mX).T@(pwc-mY)); d=np.sign(np.linalg.det(U@Vt))
    R=U@np.diag([1,1,d])@Vt; al=(R@(est-mX).T).T+mY; ate=np.sqrt(((al-pwc)**2).sum(1))
    print("median rot_err %.3f deg | %d/%d degenerate (>5deg) | ATE rmse %.3f m over %.2f m"%(
        np.nanmedian(re),int(np.sum(re>5)),len(re),np.sqrt((ate**2).mean()),
        np.linalg.norm(np.diff(pwc,axis=0),axis=1).sum()))
    return al,pwc,re,ate

if __name__=="__main__":
    run()
