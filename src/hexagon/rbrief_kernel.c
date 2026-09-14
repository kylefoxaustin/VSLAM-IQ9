/* rBRIEF: rotated 256-bit BRIEF descriptor on the Hexagon cDSP (v73/HVX). Original work (VSLAM-IQ9).
 * For each keypoint (cx,cy) with orientation angle (milliradians), sample 256 point-pairs from a
 * fixed pattern rotated by the angle; bit k = I(pa_k) < I(pb_k); pack to a 32-byte descriptor.
 * Chains the real ORB pipeline: orientation angles -> rotated descriptor.
 * Modes: 0=scalar, 1=HVX (vectorized rotations + scalar gather), 2=HVX multi-thread (keypoint bands).
 *
 * NOTE: PAT is a DETERMINISTIC SYNTHETIC pattern (LCG; 256 pairs, coords in [-15,15]) -- the same
 * SHAPE as ORB's learned bit_pattern_31_, used for performance + HVX-vs-scalar characterization.
 * It is NOT bit-compatible with OpenCV descriptors; swap in bit_pattern_31_ for interoperability.
 *
 * FINDINGS (on-DSP, EuRoC, 992 keypoints, DSP pinned TURBO_PLUS, gated BIT-EXACT to scalar):
 *   scalar 166.9 ms -> HVX 6.72 ms (24.8x) -> HVX-MT 2.70 ms (61.8x). Threading 2.49x.
 *   Scalar reference independently verified bit-exact vs a numpy rBRIEF (Hamming 0 over all 992 kp).
 *
 *   PREDICTION REFUTED: rBRIEF looks gather-bound (512 scattered byte loads / keypoint), which would
 *   predict a small HVX win. Measured, it is ROUNDING/COMPUTE-bound -- the scalar wall is the 1024
 *   software-float rounding ops / keypoint (the pattern rotations), which vectorize beautifully
 *   (24.8x, larger than orient's 3.4x or Harris's 9.25x); the scattered loads are cheap. Lesson:
 *   classify a kernel by MEASURING which op is the scalar bottleneck, not by its scariest-looking op.
 *
 *   Thread-scaling 2.49x -- same as sparse orient/Harris, vs dense-row FAST's ~3x. On this cDSP, any
 *   per-keypoint banding or scalar tail (here the scatter-gather) caps threading near 2.5x.
 *
 * HVX gotcha (probed, not guessed): Q6_Vw_equals_Vsf TRUNCATES toward zero, but scalar lrintf rounds
 * to nearest -> ~10% of sampled offsets land on the wrong pixel (mean Hamming 26/256). Fix = add a
 * sign-biased 0.5 in the float domain before the truncating convert (round_sf_to_w) -> bit-exact.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include "HAP_perf.h"
#include "worker_pool.h"
#include <math.h>

#define RB_N 256
static signed char PAT[RB_N][4] __attribute__((aligned(128))); /* ax,ay,bx,by */
static int rb_init=0;
static void rbrief_init(void){ if(rb_init)return; unsigned int s=0x1234567u;
  for(int k=0;k<RB_N;k++) for(int j=0;j<4;j++){ s=s*1103515245u+12345u; int v=(int)((s>>16)%31)-15; PAT[k][j]=(signed char)v; }
  rb_init=1; }

static void rbrief_scalar_range(const unsigned char* img, unsigned int w, const unsigned int* kps,
                                const int* ang, unsigned int i0, unsigned int i1, unsigned char* desc){
  for(unsigned int i=i0;i<i1;i++){ int cx=(int)kps[2*i], cy=(int)kps[2*i+1];
    float th=(float)ang[i]*0.001f, c=cosf(th), sn=sinf(th);
    unsigned char* d=&desc[i*32]; for(int b=0;b<32;b++) d[b]=0;
    for(int k=0;k<RB_N;k++){ int ax=PAT[k][0],ay=PAT[k][1],bx=PAT[k][2],by=PAT[k][3];
      int rax=(int)lrintf(ax*c-ay*sn), ray=(int)lrintf(ax*sn+ay*c);
      int rbx=(int)lrintf(bx*c-by*sn), rby=(int)lrintf(bx*sn+by*c);
      unsigned char pa=img[(cy+ray)*w + cx+rax], pb=img[(cy+rby)*w + cx+rbx];
      if(pa<pb) d[k>>3]|=(unsigned char)(1<<(k&7));
    }
  }
}

/* HVX: pattern coords as float so the 256 rotations vectorize (the scalar-rounding bottleneck). */
static float PAXF[RB_N] __attribute__((aligned(128))), PAYF[RB_N] __attribute__((aligned(128)));
static float PBXF[RB_N] __attribute__((aligned(128))), PBYF[RB_N] __attribute__((aligned(128)));
static void rbrief_fpat_init(void){ rbrief_init();
  for(int k=0;k<RB_N;k++){ PAXF[k]=(float)PAT[k][0]; PAYF[k]=(float)PAT[k][1]; PBXF[k]=(float)PAT[k][2]; PBYF[k]=(float)PAT[k][3]; } }
/* Q6_Vw_equals_Vsf truncates toward zero; match lrintf (round-to-nearest) via a sign-biased 0.5. */
static inline HVX_Vector round_sf_to_w(HVX_Vector tsf){
  HVX_Vector sign=Q6_V_vand_VV(tsf,Q6_V_vsplat_R(0x80000000));
  HVX_Vector half=Q6_V_vor_VV(sign,Q6_V_vsplat_R(0x3F000000)); /* +/-0.5 with tsf's sign */
  return Q6_Vw_equals_Vsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(tsf,half)));
}
static inline HVX_Vector rot_sub(HVX_Vector p,HVX_Vector q,HVX_Vector cs,HVX_Vector sn){ /* round(p*cs - q*sn) */
  return round_sf_to_w(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(p,cs),Q6_Vqf32_vmpy_VsfVsf(q,sn))));
}
static inline HVX_Vector rot_add(HVX_Vector p,HVX_Vector q,HVX_Vector cs,HVX_Vector sn){ /* round(p*cs + q*sn) */
  return round_sf_to_w(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(p,cs),Q6_Vqf32_vmpy_VsfVsf(q,sn))));
}
static void rbrief_hvx_range(const unsigned char* img, unsigned int w, const unsigned int* kps,
                             const int* ang, unsigned int i0, unsigned int i1, unsigned char* desc){
  int rax[RB_N],ray[RB_N],rbx[RB_N],rby[RB_N];
  for(unsigned int i=i0;i<i1;i++){ int cx=(int)kps[2*i], cy=(int)kps[2*i+1];
    float th=(float)ang[i]*0.001f, c=cosf(th), s=sinf(th);
    HVX_Vector cv=Q6_V_vsplat_R(*(int*)&c), sv=Q6_V_vsplat_R(*(int*)&s);
    for(int g=0;g<RB_N;g+=32){
      HVX_Vector ax=*(HVX_UVector*)&PAXF[g], ay=*(HVX_UVector*)&PAYF[g], bx=*(HVX_UVector*)&PBXF[g], by=*(HVX_UVector*)&PBYF[g];
      *(HVX_UVector*)&rax[g]=rot_sub(ax,ay,cv,sv);  /* ax*c - ay*s */
      *(HVX_UVector*)&ray[g]=rot_add(ax,ay,sv,cv);  /* ax*s + ay*c */
      *(HVX_UVector*)&rbx[g]=rot_sub(bx,by,cv,sv);  /* bx*c - by*s */
      *(HVX_UVector*)&rby[g]=rot_add(bx,by,sv,cv);  /* bx*s + by*c */
    }
    unsigned char* d=&desc[i*32]; for(int b=0;b<32;b++) d[b]=0;
    for(int k=0;k<RB_N;k++){ unsigned char pa=img[(cy+ray[k])*w + cx+rax[k]], pb=img[(cy+rby[k])*w + cx+rbx[k]];
      if(pa<pb) d[k>>3]|=(unsigned char)(1<<(k&7)); }
  }
}
typedef struct { const unsigned char* img; unsigned int w,i0,i1; const unsigned int* kps; const int* ang; unsigned char* desc; worker_synctoken_t* tok; } rbrief_band_t;
static void rbrief_band_fn(void* pv){ rbrief_band_t* b=(rbrief_band_t*)pv;
  rbrief_hvx_range(b->img,b->w,b->kps,b->ang,b->i0,b->i1,b->desc); worker_pool_synctoken_jobdone(b->tok); }

int calculator_rbrief(remote_handle64 h, const unsigned char* img, int imgLen,
                      unsigned int w, unsigned int hh, const unsigned int* kps, int kpsLen,
                      unsigned int nkp, const int* ang, int angLen, unsigned int mode,
                      unsigned char* desc, int descLen, unsigned int* us){
  unsigned long long _t0=HAP_perf_get_time_us(); rbrief_fpat_init();
  if(mode==2){ /* HVX multi-thread: band keypoints across cDSP HW threads (disjoint desc slots) */
    worker_pool_context_t ctx; worker_pool_init(&ctx);
    unsigned int nw=num_workers; if(nw<1)nw=1; if(nw>16)nw=16; if(nkp&&nw>nkp)nw=nkp;
    worker_synctoken_t tok; worker_pool_synctoken_init(&tok,nw);
    rbrief_band_t bd[16]; unsigned int per=(nkp+nw-1)/nw;
    for(unsigned int k=0;k<nw;k++){ unsigned int a=k*per,b=a+per; if(a>nkp)a=nkp; if(b>nkp)b=nkp;
      bd[k].img=img;bd[k].w=w;bd[k].kps=kps;bd[k].ang=ang;bd[k].desc=desc;bd[k].i0=a;bd[k].i1=b;bd[k].tok=&tok;
      worker_pool_job_t j; j.fptr=rbrief_band_fn; j.dptr=&bd[k]; worker_pool_submit(ctx,j); }
    worker_pool_synctoken_wait(&tok); worker_pool_deinit(&ctx);
    *us=(unsigned int)(HAP_perf_get_time_us()-_t0); return 0;
  }
  if(mode==1) rbrief_hvx_range(img,w,kps,ang,0,nkp,desc);   /* HVX offsets + scalar gather */
  else        rbrief_scalar_range(img,w,kps,ang,0,nkp,desc);/* mode 0 = scalar */
  *us=(unsigned int)(HAP_perf_get_time_us()-_t0);
  return 0;
}
