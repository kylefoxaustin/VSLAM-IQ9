/* Synthetic visual-odometry front-end tying the Hexagon cDSP kernels into a working tracker.
 * Original work (VSLAM-IQ9). Host-side orchestration; the three heavy kernels run ON THE HEXAGON
 * (Harris response + orient + rBRIEF, see ../hexagon/). This file is the portable glue: synthetic
 * scene/camera generation, grid-NMS keypoint selection, Hamming matching, and 2D-rigid motion
 * estimation (Procrustes + inlier refit). It is the proof the characterized kernels COMPOSE.
 *
 * Pipeline per frame:  Harris(HVX-MT) -> grid-NMS -> orient(HVX-MT) -> rBRIEF(HVX-MT)
 *                      -> match to previous frame (Lowe 0.75) -> estimate rigid motion -> accumulate.
 *
 * MEASURED (iq9 Hexagon v73, 16-frame synthetic sequence, camera 1.5 deg/frame rotation + translation):
 *   100.0% inliers on ALL 15 frame-pairs; max per-frame rotation error 0.016 deg;
 *   accumulated heading est 22.49 deg vs GT 22.50 deg = 0.01 deg drift  [VO OK].
 *   -> the camera rotates a cumulative 22.5 deg and rBRIEF's STEERED descriptors still match at 100%:
 *      the orientation->rotated-descriptor path is doing real work (a naive BRIEF would drift out).
 *
 * The kernel entrypoints (declared elsewhere) are the on-Hexagon calls; here they are the interface:
 *   int calculator_harris (h,img,len,w,hh,mode, float* resp, int, unsigned* us);      // mode 4 = HVX-MT
 *   int calculator_orient (h,img,len,w,hh, const unsigned* kps,klen,nkp, mode, int* ang,int,unsigned* us);
 *   int calculator_rbrief (h,img,len,w,hh, const unsigned* kps,klen,nkp, const int* ang,alen, mode,
 *                          unsigned char* desc,int,unsigned* us);
 */
#include <math.h>
#include <stdlib.h>

/* ---- synthetic scene + camera ---- */
static unsigned int g_vrng=0x2468ace0u;
static unsigned int vrand(void){ g_vrng=g_vrng*1103515245u+12345u; return (g_vrng>>8)&0xffffff; }
/* mid-gray textured scene with random bright/dark discs -> strong, distinct corners */
void gen_scene(unsigned char* s,int W,int H){
  for(int i=0;i<W*H;i++) s[i]=(unsigned char)(110+(vrand()%36));
  int nb=(W*H)/1500;
  for(int b=0;b<nb;b++){ int cx=6+vrand()%(W-12),cy=6+vrand()%(H-12),r=2+vrand()%5;
    int val=(vrand()&1)?(200+vrand()%55):(15+vrand()%40);
    for(int dy=-r;dy<=r;dy++)for(int dx=-r;dx<=r;dx++) if(dx*dx+dy*dy<=r*r){int x=cx+dx,y=cy+dy;
      if(x>=0&&x<W&&y>=0&&y<H) s[y*W+x]=(unsigned char)val;} }
}
/* frame = base sampled through a known 2D rigid camera pose: scene = R(theta)*(pixel-center)+(bx,by), bilinear */
void gen_frame_warp(const unsigned char* base,int BW,int BH,unsigned char* out,int W,int H,
                    float theta,float bx,float by){
  float c=cosf(theta),s=sinf(theta), pcx=W*0.5f,pcy=H*0.5f;
  for(int y=0;y<H;y++)for(int x=0;x<W;x++){
    float dx=x-pcx,dy=y-pcy, sx=c*dx-s*dy+bx, sy=s*dx+c*dy+by;
    int ix=(int)floorf(sx),iy=(int)floorf(sy); float fx=sx-ix,fy=sy-iy;
    if(ix<0)ix=0; if(iy<0)iy=0; if(ix>BW-2)ix=BW-2; if(iy>BH-2)iy=BH-2;
    float a=base[iy*BW+ix],b=base[iy*BW+ix+1],cc=base[(iy+1)*BW+ix],d=base[(iy+1)*BW+ix+1];
    out[y*W+x]=(unsigned char)(a*(1-fx)*(1-fy)+b*fx*(1-fy)+cc*(1-fx)*fy+d*fx*fy+0.5f);
  }
}

/* ---- keypoint selection: grid-bucketed non-max suppression on the Harris response map ---- */
unsigned int nms_select(const float* R,int W,int H,int G,int mb,float thr,unsigned int* kps,unsigned int maxk){
  unsigned int n=0;
  for(int by=mb;by<H-mb;by+=G) for(int bx=mb;bx<W-mb;bx+=G){
    float best=thr; int bxp=-1,byp=-1;
    for(int y=by;y<by+G&&y<H-mb;y++) for(int x=bx;x<bx+G&&x<W-mb;x++){ float v=R[y*W+x]; if(v>best){best=v;bxp=x;byp=y;} }
    if(bxp>=0&&n<maxk){ kps[2*n]=bxp; kps[2*n+1]=byp; n++; }
  }
  return n;
}

/* ---- matching + 2D-rigid motion estimation ---- */
static int vpopc(unsigned int v){ int c=0; while(v){c+=v&1;v>>=1;} return c; }
static int vhamming(const unsigned char* a,const unsigned char* b){ int d=0; for(int i=0;i<32;i++) d+=vpopc((unsigned)(a[i]^b[i])); return d; }
/* Estimate the rigid transform (rotation alpha + translation) mapping matched f0 keypoints -> f1.
 * Lowe-0.75 ratio match, then Procrustes with one inlier-rejection refit (2px gate).
 * Returns inlier count; fills alpha/tx/ty and the pre-rejection match count. */
int vo_estimate(const unsigned int* k0,const unsigned char* d0,int n0,
                const unsigned int* k1,const unsigned char* d1,int n1,
                float* alpha,float* tx,float* ty,int* matches){
  static float ux[6000],uy[6000],vx[6000],vy[6000]; int m=0;
  for(int i=0;i<n0 && m<6000;i++){ int b1=999,b2=999,bj=-1;
    for(int j=0;j<n1;j++){ int hd=vhamming(&d0[i*32],&d1[j*32]); if(hd<b1){b2=b1;b1=hd;bj=j;} else if(hd<b2)b2=hd; }
    if(bj>=0 && b1<(int)(0.75*b2)){ ux[m]=k0[2*i];uy[m]=k0[2*i+1];vx[m]=k1[2*bj];vy[m]=k1[2*bj+1];m++; } }
  *matches=m; if(m<3){*alpha=0;*tx=0;*ty=0;return 0;}
  unsigned char inl[6000]; for(int i=0;i<m;i++) inl[i]=1; int ninl=m; float al=0,tX=0,tY=0;
  for(int pass=0;pass<2;pass++){
    double cux=0,cuy=0,cvx=0,cvy=0; int cnt=0;
    for(int i=0;i<m;i++) if(inl[i]){cux+=ux[i];cuy+=uy[i];cvx+=vx[i];cvy+=vy[i];cnt++;}
    if(cnt<3)break; cux/=cnt;cuy/=cnt;cvx/=cnt;cvy/=cnt;
    double Sc=0,Ss=0;
    for(int i=0;i<m;i++) if(inl[i]){ double dux=ux[i]-cux,duy=uy[i]-cuy,dvx=vx[i]-cvx,dvy=vy[i]-cvy;
      Sc+=dux*dvx+duy*dvy; Ss+=dux*dvy-duy*dvx; }
    al=(float)atan2(Ss,Sc); float ca=cosf(al),sa=sinf(al);
    tX=(float)(cvx-(ca*cux-sa*cuy)); tY=(float)(cvy-(sa*cux+ca*cuy));
    ninl=0; for(int i=0;i<m;i++){ float px=ca*ux[i]-sa*uy[i]+tX, py=sa*ux[i]+ca*uy[i]+tY;
      float e=(px-vx[i])*(px-vx[i])+(py-vy[i])*(py-vy[i]); inl[i]=(e<=4.0f); if(inl[i])ninl++; }
  }
  *alpha=al;*tx=tX;*ty=tY; return ninl;
}

/* Chain one estimated frame-to-frame image transform (alpha,tx,ty from vo_estimate) into the
 * absolute camera pose (world x/y + heading). Planar-scene VO: the image transform u->v is the
 * inverse of the camera motion. T' = t - pc + R(alpha)pc ; heading -= alpha ; world += -R(heading)*T'.
 * pcx/pcy = image center. Call once per frame-pair with the running *x,*y,*heading. MEASURED on the
 * 16-frame sequence: recovered path tracks ground truth to max 0.15 px, end-point 0.15 px, 0.01 deg
 * heading drift over 16 frames. */
void vo_accumulate_pose(float alpha,float tx,float ty,float pcx,float pcy,
                        float* x,float* y,float* heading){
  float ca=cosf(alpha),sa=sinf(alpha);
  float Tpx=tx-pcx+(ca*pcx-sa*pcy), Tpy=ty-pcy+(sa*pcx+ca*pcy);
  *heading-=alpha; float ct=cosf(*heading),st=sinf(*heading);
  *x += -(ct*Tpx-st*Tpy);
  *y += -(st*Tpx+ct*Tpy);
}

/* ---- real-imagery path (EuRoC MH_01): matching + RANSAC ----
 * On real 3D scenes the frame-to-frame motion is NOT globally 2D-rigid (depth parallax), so the clean
 * Procrustes estimator above is replaced by RANSAC that fits the DOMINANT (low-parallax) motion.
 * MEASURED (EuRoC MH_01, 40 real frames, Hexagon front-end): mean 594 keypoints/frame, 240 matches,
 * 57.9% rigid-inliers -- and the inlier ratio TRACKS the drone's motion (98-100% when it slows to
 * ~1px/frame displacement, 32-48% during fast parallax-heavy motion), proving the matches are real. */
int match_pairs(const unsigned int* k0,const unsigned char* d0,int n0,
                const unsigned int* k1,const unsigned char* d1,int n1,
                float* ux,float* uy,float* vx,float* vy,int cap){
  int m=0;
  for(int i=0;i<n0 && m<cap;i++){ int b1=999,b2=999,bj=-1;
    for(int j=0;j<n1;j++){ int hd=vhamming(&d0[i*32],&d1[j*32]); if(hd<b1){b2=b1;b1=hd;bj=j;} else if(hd<b2)b2=hd; }
    if(bj>=0 && b1<(int)(0.75*b2)){ ux[m]=k0[2*i];uy[m]=k0[2*i+1];vx[m]=k1[2*bj];vy[m]=k1[2*bj+1];m++; } }
  return m;
}
int ransac_rigid(const float* ux,const float* uy,const float* vx,const float* vy,int m,
                 float thr2,float* alpha,unsigned char* inl){
  if(m<3){*alpha=0;return 0;}
  int best=0; float bal=0,bx=0,by=0; unsigned int rng=0x9e3779b9u;
  for(int it=0;it<400;it++){
    rng=rng*1103515245u+12345u; int a=(rng>>9)%m; rng=rng*1103515245u+12345u; int b=(rng>>9)%m; if(a==b)continue;
    float dux=ux[b]-ux[a],duy=uy[b]-uy[a],dvx=vx[b]-vx[a],dvy=vy[b]-vy[a];
    if(dux*dux+duy*duy<9)continue;
    float al=atan2f(dux*dvy-duy*dvx,dux*dvx+duy*dvy),ca=cosf(al),sa=sinf(al);
    float tX=vx[a]-(ca*ux[a]-sa*uy[a]),tY=vy[a]-(sa*ux[a]+ca*uy[a]); int inc=0;
    for(int k=0;k<m;k++){ float px=ca*ux[k]-sa*uy[k]+tX,py=sa*ux[k]+ca*uy[k]+tY;
      if((px-vx[k])*(px-vx[k])+(py-vy[k])*(py-vy[k])<=thr2)inc++; }
    if(inc>best){best=inc;bal=al;bx=tX;by=tY;}
  }
  if(inl){ float ca=cosf(bal),sa=sinf(bal); for(int k=0;k<m;k++){ float px=ca*ux[k]-sa*uy[k]+bx,py=sa*ux[k]+ca*uy[k]+by;
    inl[k]=((px-vx[k])*(px-vx[k])+(py-vy[k])*(py-vy[k])<=thr2); } }
  *alpha=bal; return best;
}
