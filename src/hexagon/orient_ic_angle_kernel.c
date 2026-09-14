/* ORB intensity-centroid orientation (IC_Angle: m10=sum u*I, m01=sum v*I over a circular
 * patch r=15, then atan2). Hexagon cDSP (v73/HVX). Original work (VSLAM-IQ9). Modes:
 * 0=scalar, 1=HVX (vrmpy-accumulate the centroid, atan2 tail), 2=HVX multi-thread
 * (worker_pool bands keypoints across cDSP HW threads), 3=centroid-only (isolates atan2).
 * FINDINGS (on-DSP, 1056 kps, EuRoC, gated bit-identical to scalar):
 *  - orient is CENTROID-COMPUTE-bound (atan2 only ~7% of scalar / ~24% of HVX) -- REVISES the
 *    earlier roofline that called it a 'scalar/atan2 bottleneck'. atan2 is a ~200us floor, not the wall.
 *  - HVX gives ~3.4x (836us vs 2833us); HVX multi-thread -> 329us = 8.6x over scalar, but only
 *    2.5x from threading (not the ~4x FAST got). Sparse per-keypoint work + atan2 float tail +
 *    dispatch overhead cap thread scaling -- consistent with the sparse-compute class.
 *  - MODEST vs FAST's 44x, because orient is compute-bound but SMALL-GATHER (per-kp 31px patch),
 *    while FAST is dense per-pixel. Offload taxonomy is 3-way:
 *    dense-compute (big HVX win) > sparse-compute (modest) > bandwidth-bound (none).
 * Uses Q6_Vw_vrmpyacc_VwVubVb (u8.s8->word) accumulating all rows into one vector -> 1 reduce/kp.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include "HAP_perf.h"
#include "worker_pool.h"   /* SDK libs/worker_pool: band keypoints across cDSP HW threads */
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
/* Fixed s8 weight tables (built once): UW = u-index, VWP = +v, VWN = -v, masked to the
 * circular patch (j<32 && |u|<=umax[v]). Shared by the single-thread and multi-thread HVX paths. */
static signed char OUW[ORB_HALF+1][128] __attribute__((aligned(128)));
static signed char OVWP[ORB_HALF+1][128] __attribute__((aligned(128)));
static signed char OVWN[ORB_HALF+1][128] __attribute__((aligned(128)));
static int o_wt_init=0;
static void orient_wt_init(void){ if(o_wt_init)return; umax_init();
  for(int v=0;v<=ORB_HALF;v++){ int d=g_umax[v]; for(int j=0;j<128;j++){ int u=j-ORB_HALF;
    int ok=(j<32 && u>=-d && u<=d); OUW[v][j]=ok?(signed char)u:0; OVWP[v][j]=ok?(signed char)v:0; OVWN[v][j]=ok?(signed char)(-v):0; } } o_wt_init=1; }
/* HVX orientation over a keypoint index range [i0,i1) -- the unit of parallel work. */
static void orient_hvx_range(const unsigned char* img, unsigned int w, const unsigned int* kps,
                             unsigned int i0, unsigned int i1, int* ang){
  for(unsigned int i=i0;i<i1;i++){ int cx=(int)kps[2*i], cy=(int)kps[2*i+1];
    HVX_Vector a10=Q6_V_vzero(), a01=Q6_V_vzero();
    a10=Q6_Vw_vrmpyacc_VwVubVb(a10, *(HVX_UVector*)&img[cy*w + cx-ORB_HALF], *(HVX_Vector*)OUW[0]);
    for(int v=1;v<=ORB_HALF;v++){
      HVX_Vector rp=*(HVX_UVector*)&img[(cy+v)*w + cx-ORB_HALF];
      HVX_Vector rm=*(HVX_UVector*)&img[(cy-v)*w + cx-ORB_HALF];
      HVX_Vector uw=*(HVX_Vector*)OUW[v];
      a10=Q6_Vw_vrmpyacc_VwVubVb(a10, rp, uw);
      a10=Q6_Vw_vrmpyacc_VwVubVb(a10, rm, uw);
      a01=Q6_Vw_vrmpyacc_VwVubVb(a01, rp, *(HVX_Vector*)OVWP[v]);
      a01=Q6_Vw_vrmpyacc_VwVubVb(a01, rm, *(HVX_Vector*)OVWN[v]);
    }
    int t10[32],t01[32]; *(HVX_UVector*)t10=a10; *(HVX_UVector*)t01=a01;
    long m10=0,m01=0; for(int q=0;q<32;q++){ m10+=t10[q]; m01+=t01[q]; }
    ang[i]=(int)(atan2f((float)m01,(float)m10)*1000.0f);
  }
}
/* worker_pool job: each band is a disjoint keypoint range writing disjoint ang[] slots -> no locks. */
typedef struct { const unsigned char* img; unsigned int w,i0,i1; const unsigned int* kps; int* ang; worker_synctoken_t* tok; } orient_band_t;
static void orient_band_fn(void* pv){ orient_band_t* b=(orient_band_t*)pv;
  orient_hvx_range(b->img,b->w,b->kps,b->i0,b->i1,b->ang); worker_pool_synctoken_jobdone(b->tok); }

int calculator_orient(remote_handle64 h, const unsigned char* img, int imgLen,
                      unsigned int w, unsigned int hh, const unsigned int* kps, int kpsLen,
                      unsigned int nkp, unsigned int mode, int* ang, int angLen, unsigned int* us){
  unsigned long long _t0=HAP_perf_get_time_us(); umax_init();
  if(mode==2){ /* HVX multi-thread: band the keypoints across cDSP HW threads */
    orient_wt_init();
    worker_pool_context_t ctx; worker_pool_init(&ctx);
    unsigned int nw=num_workers; if(nw<1)nw=1; if(nw>16)nw=16; if(nkp&&nw>nkp)nw=nkp;
    worker_synctoken_t tok; worker_pool_synctoken_init(&tok,nw);
    orient_band_t bd[16]; unsigned int per=(nkp+nw-1)/nw;
    for(unsigned int k=0;k<nw;k++){ unsigned int a=k*per,b=a+per; if(a>nkp)a=nkp; if(b>nkp)b=nkp;
      bd[k].img=img;bd[k].w=w;bd[k].kps=kps;bd[k].i0=a;bd[k].i1=b;bd[k].ang=ang;bd[k].tok=&tok;
      worker_pool_job_t j; j.fptr=orient_band_fn; j.dptr=&bd[k]; worker_pool_submit(ctx,j); }
    worker_pool_synctoken_wait(&tok); worker_pool_deinit(&ctx);
    *us=(unsigned int)(HAP_perf_get_time_us()-_t0); return 0;
  }
  if(mode==1){ /* HVX single-thread: vrmpy-accumulate centroid (u8 . s8) -> 1 reduce/kp, then atan2 */
    orient_wt_init(); orient_hvx_range(img,w,kps,0,nkp,ang);
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
