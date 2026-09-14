/* Dense Harris corner response on the Hexagon cDSP (v73/HVX). Original work (VSLAM-IQ9).
 * Sobel 3x3 gradients -> Ixx/Iyy/Ixy -> 3x3 box-sum -> R = det(M) - k*trace(M)^2 (k=0.04,
 * ORB's choice; no sqrt). Dense per-pixel Shi-Tomasi/Harris is VINS-Mono's real front-end
 * (goodFeaturesToTrack), so this is both realistic AND the clean dense-compute taxonomy probe.
 * Modes: 0=scalar, 1=HVX single-thread, 2=HVX multi-thread (worker_pool row bands).
 *
 * FINDINGS (on-DSP, EuRoC 752x480, DSP pinned TURBO_PLUS, gated CORNER-ranking-identical to scalar):
 *   scalar 45.3 ms -> HVX 4.90 ms (9.25x) -> HVX-MT 1.97 ms (23.1x).
 *   Gate = the metric a corner-response kernel is FOR: top-500/2000/5000 corner overlap 100.00%,
 *   all 83087 |R|>1e9 corners match to max_rel 5.4e-5 (float-exact). Raw-float outliers (~3e-2)
 *   are det-k*tr^2 cancellation in the near-zero non-corner noise floor -- irrelevant to detection.
 *
 *   THREAD-SCALING: HVX-MT threads only 2.49x -- like sparse orient (2.5x), NOT like dense FAST (3x).
 *   De-confounded (vectorizing the serial zero didn't change it): the cause is THIS implementation
 *   being load-heavy (108 unaligned u8-vector loads / 64 output px, because gradients are recomputed
 *   for all 9 box positions), pushing it toward bandwidth-bound. => thread-scaling is kernel x
 *   IMPLEMENTATION arithmetic-intensity, not the abstract kernel class alone. An optimized Harris
 *   (each gradient computed once, reused via a separable horizontal+vertical box-sum) should thread
 *   closer to 3x. Compare cross-DSP by MEASURED arithmetic intensity, not by kernel name.
 *
 * HVX note: u8->int widening on HVX DEINTERLEAVES (probed: a 0..127 ramp comes back stride-4).
 * In-order widen = Q6_Wuh_vzxt_Vub then Q6_W_vshuff_VVR(hi,lo,-2) (probed identity). This kernel
 * dodges the issue entirely: horizontal neighbors come from unaligned u8 loads (in-order), never a
 * post-widen lane shift; the square's even/odd int32 split IS the lo/hi pixel partition, reunited
 * with a 32-bit shuff (-4) at store.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include "HAP_perf.h"
#include "worker_pool.h"
#include <math.h>
#include <stdlib.h>

#define HARRIS_K 0.04f

/* scalar reference (also used for the border/tail columns of the HVX path) */
static float harris_pixel(const unsigned char* img, unsigned int w, unsigned int x, unsigned int y){
  long Sxx=0,Syy=0,Sxy=0;
  for(int by=-1;by<=1;by++) for(int bx=-1;bx<=1;bx++){
    const unsigned char* p=&img[(y+by)*w + (x+bx)];
    int gx = (int)p[-(int)w+1]+2*(int)p[1]+(int)p[w+1] - (int)p[-(int)w-1]-2*(int)p[-1]-(int)p[w-1];
    int gy = (int)p[w-1]+2*(int)p[w]+(int)p[w+1] - (int)p[-(int)w-1]-2*(int)p[-(int)w]-(int)p[-(int)w+1];
    Sxx += (long)gx*gx; Syy += (long)gy*gy; Sxy += (long)gx*gy;
  }
  float fxx=(float)Sxx, fyy=(float)Syy, fxy=(float)Sxy;
  float det=fxx*fyy - fxy*fxy, tr=fxx+fyy;
  return det - HARRIS_K*tr*tr;
}
static void harris_scalar_rows(const unsigned char* img, unsigned int w, unsigned int hh,
                               float* resp, unsigned int y0, unsigned int y1){
  for(unsigned int y=y0;y<y1;y++)
    for(unsigned int x=2;x+2<w;x++) resp[y*w+x]=harris_pixel(img,w,x,y);
}

/* load 128 u8 at a (possibly unaligned) base, return the first 64 pixels as IN-ORDER u16 */
static inline HVX_Vector ld64_u16(const unsigned char* base){
  HVX_VectorPair h=Q6_Wuh_vzxt_Vub(*(HVX_UVector*)base);
  return Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(h),Q6_V_lo_W(h),-2));
}
static inline HVX_Vector vh_smooth(HVX_Vector a,HVX_Vector b,HVX_Vector c){ /* a + 2b + c (int16) */
  return Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(a,c),Q6_Vh_vadd_VhVh(b,b));
}
/* per-half (32 pixels) Harris response from int32 Sxx/Syy/Sxy, in HVX float */
static inline HVX_Vector harris_resp_half(HVX_Vector Sxx,HVX_Vector Syy,HVX_Vector Sxy){
  HVX_Vector fxx=Q6_Vsf_equals_Vw(Sxx), fyy=Q6_Vsf_equals_Vw(Syy), fxy=Q6_Vsf_equals_Vw(Sxy);
  HVX_Vector det=Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(fxx,fyy),Q6_Vqf32_vmpy_VsfVsf(fxy,fxy));
  HVX_Vector tr=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(fxx,fyy));
  HVX_Vector trsq=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(tr,tr));
  HVX_Vector ksf=Q6_V_vsplat_R(0x3D23D70A); /* 0.04f */
  HVX_Vector ktr=Q6_Vqf32_vmpy_VsfVsf(trsq,ksf);
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(det,ktr)); /* det - k*tr^2 */
}
static void harris_hvx_rows(const unsigned char* img, unsigned int w, unsigned int hh,
                            float* resp, unsigned int y0, unsigned int y1){
  for(unsigned int y=y0;y<y1;y++){
    unsigned int x0=2;
    for(; x0+64+2<=w; x0+=64){
      HVX_Vector Sxl=Q6_V_vzero(),Sxh=Q6_V_vzero(),Syl=Q6_V_vzero(),Syh=Q6_V_vzero(),Sil=Q6_V_vzero(),Sih=Q6_V_vzero();
      for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++){
        const unsigned char* rm=&img[((int)y+dy-1)*(int)w + (int)x0+dx];
        const unsigned char* rc=&img[((int)y+dy)*(int)w   + (int)x0+dx];
        const unsigned char* rp=&img[((int)y+dy+1)*(int)w + (int)x0+dx];
        HVX_Vector sR=vh_smooth(ld64_u16(rm+1),ld64_u16(rc+1),ld64_u16(rp+1));
        HVX_Vector sL=vh_smooth(ld64_u16(rm-1),ld64_u16(rc-1),ld64_u16(rp-1));
        HVX_Vector gx=Q6_Vh_vsub_VhVh(sR,sL);
        HVX_Vector hp=vh_smooth(ld64_u16(rp-1),ld64_u16(rp),ld64_u16(rp+1));
        HVX_Vector hm=vh_smooth(ld64_u16(rm-1),ld64_u16(rm),ld64_u16(rm+1));
        HVX_Vector gy=Q6_Vh_vsub_VhVh(hp,hm);
        HVX_VectorPair xx=Q6_Ww_vmpy_VhVh(gx,gx), yy=Q6_Ww_vmpy_VhVh(gy,gy), xy=Q6_Ww_vmpy_VhVh(gx,gy);
        Sxl=Q6_Vw_vadd_VwVw(Sxl,Q6_V_lo_W(xx)); Sxh=Q6_Vw_vadd_VwVw(Sxh,Q6_V_hi_W(xx));
        Syl=Q6_Vw_vadd_VwVw(Syl,Q6_V_lo_W(yy)); Syh=Q6_Vw_vadd_VwVw(Syh,Q6_V_hi_W(yy));
        Sil=Q6_Vw_vadd_VwVw(Sil,Q6_V_lo_W(xy)); Sih=Q6_Vw_vadd_VwVw(Sih,Q6_V_hi_W(xy));
      }
      HVX_Vector Rlo=harris_resp_half(Sxl,Syl,Sil); /* even pixels 0,2,.. */
      HVX_Vector Rhi=harris_resp_half(Sxh,Syh,Sih); /* odd pixels 1,3,.. */
      HVX_VectorPair Ro=Q6_W_vshuff_VVR(Rhi,Rlo,-4);/* interleave -> in-order 64 floats */
      *(HVX_UVector*)&resp[y*w+x0]    = Q6_V_lo_W(Ro);
      *(HVX_UVector*)&resp[y*w+x0+32] = Q6_V_hi_W(Ro);
    }
    for(unsigned int x=x0; x+2<w; x++) resp[y*w+x]=harris_pixel(img,w,x,y); /* tail cols */
  }
}
typedef struct { const unsigned char* img; unsigned int w,hh,y0,y1; float* resp; worker_synctoken_t* tok; } harris_band_t;
static void harris_band_fn(void* pv){ harris_band_t* b=(harris_band_t*)pv;
  harris_hvx_rows(b->img,b->w,b->hh,b->resp,b->y0,b->y1); worker_pool_synctoken_jobdone(b->tok); }

/* ---- OPTIMIZED single-pass Harris (modes 3/4): each pixel's gradient-products computed ONCE into
 * rolling in-order int32 scratch rows, then a 3x3 box-sum reads scratch (no per-box gradient recompute).
 * RESULT: faster absolute (OPT-MT 1.24 ms = 36.6x scalar, the best Harris variant) BUT threads WORSE
 * (2.30x vs the naive 2.48x) -- the prediction that "fewer loads -> threads closer to 3x" is REFUTED.
 * Mechanism: computing gradients once needs a SCRATCH ROUND-TRIP through the shared L2/DDR path, and
 * the cDSP HW threads share that path -- so caching intermediates converts arithmetic into
 * shared-memory traffic, which is exactly what caps HVX thread-scaling. Thread-scaling tracks pressure
 * on the SHARED MEMORY PATH, not the kernel's compute/memory ratio in isolation. ---- */
static void grad_row(const unsigned char* img, unsigned int w, unsigned int Y, int* ixx,int* iyy,int* ixy){
  unsigned int x0=2;
  for(; x0+64+1<=w; x0+=64){
    const unsigned char* rm=&img[(Y-1)*w+x0],*rc=&img[Y*w+x0],*rp=&img[(Y+1)*w+x0];
    HVX_Vector gx=Q6_Vh_vsub_VhVh(vh_smooth(ld64_u16(rm+1),ld64_u16(rc+1),ld64_u16(rp+1)),
                                  vh_smooth(ld64_u16(rm-1),ld64_u16(rc-1),ld64_u16(rp-1)));
    HVX_Vector gy=Q6_Vh_vsub_VhVh(vh_smooth(ld64_u16(rp-1),ld64_u16(rp),ld64_u16(rp+1)),
                                  vh_smooth(ld64_u16(rm-1),ld64_u16(rm),ld64_u16(rm+1)));
    HVX_VectorPair xx=Q6_Ww_vmpy_VhVh(gx,gx),yy=Q6_Ww_vmpy_VhVh(gy,gy),xy=Q6_Ww_vmpy_VhVh(gx,gy);
    HVX_VectorPair xo=Q6_W_vshuff_VVR(Q6_V_hi_W(xx),Q6_V_lo_W(xx),-4); /* -> in-order int32 */
    HVX_VectorPair yo=Q6_W_vshuff_VVR(Q6_V_hi_W(yy),Q6_V_lo_W(yy),-4);
    HVX_VectorPair io=Q6_W_vshuff_VVR(Q6_V_hi_W(xy),Q6_V_lo_W(xy),-4);
    *(HVX_UVector*)&ixx[x0]=Q6_V_lo_W(xo); *(HVX_UVector*)&ixx[x0+32]=Q6_V_hi_W(xo);
    *(HVX_UVector*)&iyy[x0]=Q6_V_lo_W(yo); *(HVX_UVector*)&iyy[x0+32]=Q6_V_hi_W(yo);
    *(HVX_UVector*)&ixy[x0]=Q6_V_lo_W(io); *(HVX_UVector*)&ixy[x0+32]=Q6_V_hi_W(io);
  }
  for(unsigned int x=x0; x+1<w; x++){ /* scalar tail cols [x0, w-1) */
    const unsigned char* p=&img[Y*w+x];
    int gx=(int)p[-(int)w+1]+2*(int)p[1]+(int)p[w+1]-(int)p[-(int)w-1]-2*(int)p[-1]-(int)p[w-1];
    int gy=(int)p[w-1]+2*(int)p[w]+(int)p[w+1]-(int)p[-(int)w-1]-2*(int)p[-(int)w]-(int)p[-(int)w+1];
    ixx[x]=gx*gx; iyy[x]=gy*gy; ixy[x]=gx*gy; }
  { const unsigned char* p=&img[Y*w+1]; /* col 1 (HVX bulk starts at col 2) */
    int gx=(int)p[-(int)w+1]+2*(int)p[1]+(int)p[w+1]-(int)p[-(int)w-1]-2*(int)p[-1]-(int)p[w-1];
    int gy=(int)p[w-1]+2*(int)p[w]+(int)p[w+1]-(int)p[-(int)w-1]-2*(int)p[-(int)w]-(int)p[-(int)w+1];
    ixx[1]=gx*gx; iyy[1]=gy*gy; ixy[1]=gx*gy; }
}
static void harris_opt_rows(const unsigned char* img, unsigned int w, unsigned int hh,
                            float* resp, unsigned int y0, unsigned int y1){
  int* buf=(int*)malloc((size_t)9*w*sizeof(int)); if(!buf) return;
  int* ixx[3]; int* iyy[3]; int* ixy[3];
  for(int k=0;k<3;k++){ ixx[k]=buf+((size_t)k*3+0)*w; iyy[k]=buf+((size_t)k*3+1)*w; ixy[k]=buf+((size_t)k*3+2)*w; }
  grad_row(img,w,y0-1,ixx[(y0-1)%3],iyy[(y0-1)%3],ixy[(y0-1)%3]);
  grad_row(img,w,y0,  ixx[y0%3],    iyy[y0%3],    ixy[y0%3]);
  for(unsigned int y=y0;y<y1;y++){
    grad_row(img,w,y+1, ixx[(y+1)%3], iyy[(y+1)%3], ixy[(y+1)%3]);
    int *xA=ixx[(y-1)%3],*xB=ixx[y%3],*xC=ixx[(y+1)%3];
    int *yA=iyy[(y-1)%3],*yB=iyy[y%3],*yC=iyy[(y+1)%3];
    int *iA=ixy[(y-1)%3],*iB=ixy[y%3],*iC=ixy[(y+1)%3];
    unsigned int x0=2;
    for(; x0+32<=w-2; x0+=32){
      HVX_Vector Sxx=Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(
          Q6_Vw_vadd_VwVw(*(HVX_UVector*)&xA[x0-1],Q6_Vw_vadd_VwVw(*(HVX_UVector*)&xA[x0],*(HVX_UVector*)&xA[x0+1])),
          Q6_Vw_vadd_VwVw(*(HVX_UVector*)&xB[x0-1],Q6_Vw_vadd_VwVw(*(HVX_UVector*)&xB[x0],*(HVX_UVector*)&xB[x0+1]))),
          Q6_Vw_vadd_VwVw(*(HVX_UVector*)&xC[x0-1],Q6_Vw_vadd_VwVw(*(HVX_UVector*)&xC[x0],*(HVX_UVector*)&xC[x0+1])));
      HVX_Vector Syy=Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(
          Q6_Vw_vadd_VwVw(*(HVX_UVector*)&yA[x0-1],Q6_Vw_vadd_VwVw(*(HVX_UVector*)&yA[x0],*(HVX_UVector*)&yA[x0+1])),
          Q6_Vw_vadd_VwVw(*(HVX_UVector*)&yB[x0-1],Q6_Vw_vadd_VwVw(*(HVX_UVector*)&yB[x0],*(HVX_UVector*)&yB[x0+1]))),
          Q6_Vw_vadd_VwVw(*(HVX_UVector*)&yC[x0-1],Q6_Vw_vadd_VwVw(*(HVX_UVector*)&yC[x0],*(HVX_UVector*)&yC[x0+1])));
      HVX_Vector Sxy=Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(
          Q6_Vw_vadd_VwVw(*(HVX_UVector*)&iA[x0-1],Q6_Vw_vadd_VwVw(*(HVX_UVector*)&iA[x0],*(HVX_UVector*)&iA[x0+1])),
          Q6_Vw_vadd_VwVw(*(HVX_UVector*)&iB[x0-1],Q6_Vw_vadd_VwVw(*(HVX_UVector*)&iB[x0],*(HVX_UVector*)&iB[x0+1]))),
          Q6_Vw_vadd_VwVw(*(HVX_UVector*)&iC[x0-1],Q6_Vw_vadd_VwVw(*(HVX_UVector*)&iC[x0],*(HVX_UVector*)&iC[x0+1])));
      *(HVX_UVector*)&resp[y*w+x0]=harris_resp_half(Sxx,Syy,Sxy);
    }
    for(unsigned int x=x0; x+2<w; x++) resp[y*w+x]=harris_pixel(img,w,x,y); /* tail cols */
  }
  free(buf);
}
static void harris_opt_band_fn(void* pv){ harris_band_t* b=(harris_band_t*)pv;
  harris_opt_rows(b->img,b->w,b->hh,b->resp,b->y0,b->y1); worker_pool_synctoken_jobdone(b->tok); }

int calculator_harris(remote_handle64 h, const unsigned char* img, int imgLen,
                      unsigned int w, unsigned int hh, unsigned int mode,
                      float* resp, int respLen, unsigned int* us){
  unsigned long long _t0=HAP_perf_get_time_us();
  { unsigned int n=w*hh,i=0; for(;i+32<=n;i+=32) *(HVX_Vector*)&resp[i]=Q6_V_vzero(); /* HVX zero */
    for(;i<n;i++) resp[i]=0.0f; }
  if(mode==2){ /* HVX multi-thread: band rows across cDSP HW threads */
    worker_pool_context_t ctx; worker_pool_init(&ctx);
    unsigned int nw=num_workers; if(nw<1)nw=1; if(nw>16)nw=16;
    unsigned int y0=2,y1=hh-2,rows=y1-y0; if(nw>rows)nw=rows?rows:1;
    worker_synctoken_t tok; worker_pool_synctoken_init(&tok,nw);
    harris_band_t bd[16]; unsigned int per=(rows+nw-1)/nw;
    for(unsigned int k=0;k<nw;k++){ unsigned int a=y0+k*per,b=a+per; if(a>y1)a=y1; if(b>y1)b=y1;
      bd[k].img=img;bd[k].w=w;bd[k].hh=hh;bd[k].y0=a;bd[k].y1=b;bd[k].resp=resp;bd[k].tok=&tok;
      worker_pool_job_t j; j.fptr=harris_band_fn; j.dptr=&bd[k]; worker_pool_submit(ctx,j); }
    worker_pool_synctoken_wait(&tok); worker_pool_deinit(&ctx);
  }
  else if(mode==4){ /* OPT single-pass HVX, multi-thread */
    worker_pool_context_t ctx; worker_pool_init(&ctx);
    unsigned int nw=num_workers; if(nw<1)nw=1; if(nw>16)nw=16;
    unsigned int y0=2,y1=hh-2,rows=y1-y0; if(nw>rows)nw=rows?rows:1;
    worker_synctoken_t tok; worker_pool_synctoken_init(&tok,nw);
    harris_band_t bd[16]; unsigned int per=(rows+nw-1)/nw;
    for(unsigned int k=0;k<nw;k++){ unsigned int a=y0+k*per,b=a+per; if(a>y1)a=y1; if(b>y1)b=y1;
      bd[k].img=img;bd[k].w=w;bd[k].hh=hh;bd[k].y0=a;bd[k].y1=b;bd[k].resp=resp;bd[k].tok=&tok;
      worker_pool_job_t j; j.fptr=harris_opt_band_fn; j.dptr=&bd[k]; worker_pool_submit(ctx,j); }
    worker_pool_synctoken_wait(&tok); worker_pool_deinit(&ctx);
  }
  else if(mode==3) harris_opt_rows(img,w,hh,resp,2,hh-2);   /* OPT single-pass HVX, 1 thread */
  else if(mode==1) harris_hvx_rows(img,w,hh,resp,2,hh-2);   /* HVX single-thread */
  else             harris_scalar_rows(img,w,hh,resp,2,hh-2);/* mode 0 = scalar */
  *us=(unsigned int)(HAP_perf_get_time_us()-_t0);
  return 0;
}
