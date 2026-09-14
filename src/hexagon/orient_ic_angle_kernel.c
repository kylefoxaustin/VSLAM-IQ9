/* ORB intensity-centroid orientation (IC_Angle: m10=sum u*I, m01=sum v*I over a circular
 * patch r=15, then atan2). Hexagon cDSP (v73/HVX). Original work (VSLAM-IQ9). Modes:
 * 0=scalar, 1=HVX (vrmpy-accumulate the centroid, atan2 tail), 3=centroid-only (isolates atan2).
 * FINDINGS (on-DSP, 1056 kps, EuRoC, gated bit-identical to scalar):
 *  - orient is CENTROID-COMPUTE-bound (atan2 only ~7% of scalar / ~24% of HVX) -- REVISES the
 *    earlier roofline that called it a 'scalar/atan2 bottleneck'. atan2 is a ~200us floor, not the wall.
 *  - HVX gives ~3.45x (823us vs 2840us) -- MODEST vs FAST's 44x, because orient is compute-bound but
 *    SMALL-GATHER (per-kp 31px patch), while FAST is dense per-pixel. Offload taxonomy is 3-way:
 *    dense-compute (big HVX win) > sparse-compute (modest) > bandwidth-bound (none).
 * Uses Q6_Vw_vrmpyacc_VwVubVb (u8.s8->word) accumulating all rows into one vector -> 1 reduce/kp.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include "HAP_perf.h"
#include <math.h>

/* ---- ORB intensity-centroid orientation (IC_Angle), circular patch radius 15 ---- */
#define ORB_HALF 15
static int g_umax[ORB_HALF+1]; static int g_umax_init=0;
static void umax_init(void){ if(g_umax_init)return; for(int v=0;v<=ORB_HALF;v++){ int u=0; while((u+1)*(u+1)+v*v<=ORB_HALF*ORB_HALF)u++; g_umax[v]=u;} g_umax_init=1; }
static long ic_angle_scalar(const unsigned char* img, unsigned int w, int cx, int cy){
  long m01=0,m10=0; const unsigned char* c=&img[cy*w+cx];
  for(int u=-ORB_HALF;u<=ORB_HALF;u++) m10 += u*(int)c[u];      /* center row v=0 */
  for(int v=1;v<=ORB_HALF;v++){ int d=g_umax[v]; long vsum=0;
    const unsigned char* rp=&img[(cy+v)*w+cx]; const unsigned char* rm=&img[(cy-v)*w+cx];
    for(int u=-d;u<=d;u++){ int vp=rp[u], vm=rm[u]; vsum += (vp-vm); m10 += u*(vp+vm); }
    m01 += (long)v*vsum;
  }
  return (long)(atan2f((float)m01,(float)m10)*1000.0f); /* milliradians */
}
int calculator_orient(remote_handle64 h, const unsigned char* img, int imgLen,
                      unsigned int w, unsigned int hh, const unsigned int* kps, int kpsLen,
                      unsigned int nkp, unsigned int mode, int* ang, int angLen, unsigned int* us){
  unsigned long long _t0=HAP_perf_get_time_us(); umax_init();
  /* mode 0 = scalar (HVX added later) */
  if(mode==1){ /* HVX: vrmpy-accumulate centroid (u8 . s8 weights) -> 1 reduce/kp, then atan2 */
    static signed char UW[ORB_HALF+1][128] __attribute__((aligned(128)));
    static signed char VWP[ORB_HALF+1][128] __attribute__((aligned(128)));
    static signed char VWN[ORB_HALF+1][128] __attribute__((aligned(128)));
    static int wi=0;
    if(!wi){ for(int v=0;v<=ORB_HALF;v++){ int d=g_umax[v]; for(int j=0;j<128;j++){ int u=j-ORB_HALF;
      int ok=(j<32 && u>=-d && u<=d); UW[v][j]=ok?(signed char)u:0; VWP[v][j]=ok?(signed char)v:0; VWN[v][j]=ok?(signed char)(-v):0; } } wi=1; }
    for(unsigned int i=0;i<nkp;i++){ int cx=(int)kps[2*i], cy=(int)kps[2*i+1];
      HVX_Vector a10=Q6_V_vzero(), a01=Q6_V_vzero();
      HVX_Vector r0=*(HVX_UVector*)&img[cy*w + cx-ORB_HALF];
      a10=Q6_Vw_vrmpyacc_VwVubVb(a10, r0, *(HVX_Vector*)UW[0]);
      for(int v=1;v<=ORB_HALF;v++){
        HVX_Vector rp=*(HVX_UVector*)&img[(cy+v)*w + cx-ORB_HALF];
        HVX_Vector rm=*(HVX_UVector*)&img[(cy-v)*w + cx-ORB_HALF];
        HVX_Vector uw=*(HVX_Vector*)UW[v];
        a10=Q6_Vw_vrmpyacc_VwVubVb(a10, rp, uw);
        a10=Q6_Vw_vrmpyacc_VwVubVb(a10, rm, uw);
        a01=Q6_Vw_vrmpyacc_VwVubVb(a01, rp, *(HVX_Vector*)VWP[v]);
        a01=Q6_Vw_vrmpyacc_VwVubVb(a01, rm, *(HVX_Vector*)VWN[v]);
      }
      int t10[32],t01[32]; *(HVX_UVector*)t10=a10; *(HVX_UVector*)t01=a01;
      long m10=0,m01=0; for(int q=0;q<32;q++){ m10+=t10[q]; m01+=t01[q]; }
      ang[i]=(int)(atan2f((float)m01,(float)m10)*1000.0f);
    }
    *us=(unsigned int)(HAP_perf_get_time_us()-_t0); return 0;
  }
  if(mode==3){ /* centroid only, NO atan2 -> isolate atan2 cost */
    for(unsigned int i=0;i<nkp;i++){ int cx=(int)kps[2*i], cy=(int)kps[2*i+1];
      long m01=0,m10=0; const unsigned char* c=&img[cy*w+cx];
      for(int u=-ORB_HALF;u<=ORB_HALF;u++) m10+=u*(int)c[u];
      for(int v=1;v<=ORB_HALF;v++){ int d=g_umax[v]; long vs=0;
        const unsigned char* rp=&img[(cy+v)*w+cx]; const unsigned char* rm=&img[(cy-v)*w+cx];
        for(int u=-d;u<=d;u++){ int vp=rp[u],vm=rm[u]; vs+=(vp-vm); m10+=u*(vp+vm);} m01+=(long)v*vs; }
      ang[i]=(int)(m10+m01); }
  } else {
    for(unsigned int i=0;i<nkp;i++){ int cx=(int)kps[2*i], cy=(int)kps[2*i+1];
      ang[i]=(int)ic_angle_scalar(img,w,cx,cy); }
  }
  *us=(unsigned int)(HAP_perf_get_time_us()-_t0);
  FARF(RUNTIME_HIGH,"DSP orient mode %u: %u kps %u us",mode,nkp,*us);
  return 0;
}
