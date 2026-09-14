/* 5x5 binomial Gaussian blur ([1,4,6,4,1] outer product / 256, replicate border) for the
 * Hexagon cDSP (v73/HVX). Original work (VSLAM-IQ9). fastRPC skel: mode 0=scalar ref,
 * 1=HVX single-thread, 2=HVX multi-thread (worker_pool). Bit-correct vs CPU ref (<=1 LSB,
 * round-vs-truncate). FINDING: this kernel is BANDWIDTH-BOUND -- HVX threads share DDR so
 * multithreading barely helps, and it LOSES to the CPU's optimized separable blur. Contrast
 * with FAST (compute-bound) which wins big on HVX+MT. Offload compute-bound kernels, not BW-bound.
 *
 * HVX narrow gotcha (root-caused via micro-test): Q6_Vub_vasr_VuhVuhR_rnd_sat saturates large
 * u16 regardless of shift, so do the >>8 in the u16 domain first (Q6_Vuh_vlsr_VuhR) then narrow.
 * And byte-replicate the vmpy scalar (wgt*0x01010101) since V6_vmpyub cycles R's 4 bytes.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include "HAP_perf.h"
#include "worker_pool.h"

/* ---- 5x5 binomial Gaussian blur (weights = [1,4,6,4,1] outer product / 256), replicate border ---- */
static const int GW[5]={1,4,6,4,1};
static unsigned int _clampu(int v,int lo,int hi){ return (unsigned)(v<lo?lo:(v>hi?hi:v)); }

static void blur_rows_scalar(const unsigned char* img, unsigned int w, unsigned int hh,
                             unsigned char* out, unsigned int y0, unsigned int y1){
  for(unsigned int y=y0;y<y1;y++) for(unsigned int x=0;x<w;x++){
    int acc=0;
    for(int dy=-2;dy<=2;dy++){ int yy=(int)_clampu((int)y+dy,0,(int)hh-1);
      for(int dx=-2;dx<=2;dx++){ int xx=(int)_clampu((int)x+dx,0,(int)w-1);
        acc += GW[dy+2]*GW[dx+2]*img[yy*w+xx]; } }
    out[y*w+x]=(unsigned char)((acc+128)>>8);
  }
}
/* HVX: interior columns [2,w-2) vectorized; edges (x<2 || x>=w-2) scalar. Rows [y0,y1). */
static void blur_rows_hvx(const unsigned char* img, unsigned int w, unsigned int hh,
                          unsigned char* out, unsigned int y0, unsigned int y1){
  const int VLEN=128;
  for(unsigned int y=y0;y<y1;y++){
    int ym[5]; for(int d=-2;d<=2;d++) ym[d+2]=(int)_clampu((int)y+d,0,(int)hh-1);
    unsigned int x=2;
    for(; x+VLEN<=w-2; x+=VLEN){
      HVX_VectorPair acc; int first=1;      /* accumulate in one u16 pair to keep vmpy interleave */
      for(int dy=0;dy<5;dy++){ const unsigned char* row=&img[ym[dy]*w];
        for(int dx=-2;dx<=2;dx++){ int wgt=GW[dy]*GW[dx+2];
          HVX_Vector v=*(HVX_UVector*)&row[x+dx];
          if(first){ acc=Q6_Wuh_vmpy_VubRub(v,(unsigned int)(wgt*0x01010101u)); first=0; }
          else       acc=Q6_Wuh_vmpyacc_WuhVubRub(acc,v,(unsigned int)(wgt*0x01010101u));
        } }
      HVX_Vector lo8=Q6_Vuh_vlsr_VuhR(Q6_V_lo_W(acc),8), hi8=Q6_Vuh_vlsr_VuhR(Q6_V_hi_W(acc),8);
      HVX_Vector o=Q6_Vub_vasr_VuhVuhR_rnd_sat(hi8,lo8,0); /* values now <=255 -> narrow */
      *(HVX_UVector*)&out[y*w+x]=o;
    }
  }
}
typedef struct { const unsigned char* img; unsigned int w,hh,y0,y1; unsigned char* out; worker_synctoken_t* tok; } blur_band_t;
static void blur_band_fn(void* pv){ blur_band_t* b=(blur_band_t*)pv; blur_rows_hvx(b->img,b->w,b->hh,b->out,b->y0,b->y1); worker_pool_synctoken_jobdone(b->tok); }

int calculator_blur(remote_handle64 h, const unsigned char* img, int imgLen,
                    unsigned int w, unsigned int hh, unsigned int mode,
                    unsigned char* out, int outLen, unsigned int* us){
  unsigned long long _t0=HAP_perf_get_time_us();
  if(mode==2){ /* multithreaded HVX blur (interior); scalar edge fixup after */
    worker_pool_context_t ctx; worker_pool_init(&ctx);
    unsigned int nw=num_workers; if(nw<1)nw=1; if(nw>16)nw=16;
    worker_synctoken_t tok; worker_pool_synctoken_init(&tok,nw);
    blur_band_t bd[16]; unsigned int per=(hh+nw-1)/nw;
    for(unsigned int i=0;i<nw;i++){ unsigned int by0=i*per,by1=by0+per; if(by0>hh)by0=hh; if(by1>hh)by1=hh;
      bd[i].img=img;bd[i].w=w;bd[i].hh=hh;bd[i].y0=by0;bd[i].y1=by1;bd[i].out=out;bd[i].tok=&tok;
      worker_pool_job_t j; j.fptr=blur_band_fn; j.dptr=&bd[i]; worker_pool_submit(ctx,j); }
    worker_pool_synctoken_wait(&tok); worker_pool_deinit(&ctx);
    const int VLEN=128; unsigned int xvend=2+((w-4)/VLEN)*VLEN;
    for(unsigned int y=0;y<hh;y++) for(unsigned int x=0;x<w;x++){ if(x>=2&&x<xvend)continue;
      int acc=0; for(int dy=-2;dy<=2;dy++){int yy=(int)_clampu((int)y+dy,0,(int)hh-1);
        for(int dx=-2;dx<=2;dx++){int xx=(int)_clampu((int)x+dx,0,(int)w-1); acc+=GW[dy+2]*GW[dx+2]*img[yy*w+xx];}}
      out[y*w+x]=(unsigned char)((acc+128)>>8); }
    *us=(unsigned int)(HAP_perf_get_time_us()-_t0); return 0;
  }
  if(mode==0){ blur_rows_scalar(img,w,hh,out,0,hh); }
  else {
    /* HVX interior for all rows; then scalar fixup for the left/right edge columns */
    blur_rows_hvx(img,w,hh,out,0,hh);
    const int VLEN=128; unsigned int xvend = 2 + ((w-4)/VLEN)*VLEN; /* last vectorized x end */
    for(unsigned int y=0;y<hh;y++){
      for(unsigned int x=0;x<w;x++){
        if(x>=2 && x<xvend) continue; /* covered by HVX */
        int acc=0; for(int dy=-2;dy<=2;dy++){ int yy=(int)_clampu((int)y+dy,0,(int)hh-1);
          for(int dx=-2;dx<=2;dx++){ int xx=(int)_clampu((int)x+dx,0,(int)w-1);
            acc+=GW[dy+2]*GW[dx+2]*img[yy*w+xx]; } }
        out[y*w+x]=(unsigned char)((acc+128)>>8);
      }
    }
  }
  *us=(unsigned int)(HAP_perf_get_time_us()-_t0);
  FARF(RUNTIME_HIGH,"DSP blur mode %u: %u us",mode,*us);
  return 0;
}
