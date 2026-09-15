/* Image-pyramid octave: 2x2 box downsample to half resolution on the Hexagon cDSP. Original work.
 * out[y][x] = (in[2y][2x] + in[2y][2x+1] + in[2y+1][2x] + in[2y+1][2x+1] + 2) >> 2.
 * Modes: 0=scalar, 1=HVX, 2=HVX multi-thread (row bands). All bit-exact (max|diff|=0).
 *
 * FINDINGS (iq9 v73, 752x480 -> 376x240, wall-clock e2e): scalar 139us, HVX 217us, HVX-MT 216us.
 *   => HVX gives NO benefit (slightly slower), threading gives NO benefit — pyramid downsample is
 *   BANDWIDTH-BOUND (reads the full image, writes 1/4, ~4 adds/pixel), exactly as the roofline
 *   predicted. On a kernel this cheap the HVX widen/narrow overhead + the scalar tail LOSE to a
 *   compiler-optimized scalar loop. Second BW-bound instance (with 5x5 blur) — the offload taxonomy's
 *   BW-bound class now has two members: keep pyramid/downsample on the CPU, don't offload it to HVX.
 *
 * Measurement note: this kernel is sub-microsecond single-shot, so the on-DSP us timer reads 0, and an
 * internal repeat loop is dead-store-eliminated + loop-invariant-hoisted (pure fn of fixed input).
 * Time it with host wall-clock (clock_gettime), min over many calls.
 *
 * HVX narrow gotcha (from the blur kernel): Q6_Vub_vasr_VuhVuhR_rnd_sat saturates large u16 regardless
 * of shift, so do the >>2 in the u16 domain FIRST (Q6_Vuh_vlsr_VuhR) then narrow at shift 0.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include "HAP_perf.h"
#include "worker_pool.h"

static void pyrdown_scalar_rows(const unsigned char* img, unsigned int w, unsigned int ow,
                                unsigned char* out, unsigned int oy0, unsigned int oy1){
  for(unsigned int oy=oy0;oy<oy1;oy++){ const unsigned char* r0=&img[(2*oy)*w],*r1=&img[(2*oy+1)*w];
    for(unsigned int ox=0;ox<ow;ox++){ unsigned int x=2*ox;
      out[oy*ow+ox]=(unsigned char)((r0[x]+r0[x+1]+r1[x]+r1[x+1]+2)>>2); } }
}
static void pyrdown_hvx_rows(const unsigned char* img, unsigned int w, unsigned int ow,
                             unsigned char* out, unsigned int oy0, unsigned int oy1){
  HVX_Vector two=Q6_V_vsplat_R(0x00020002); /* +2 per u16 lane, round-to-nearest */
  for(unsigned int oy=oy0;oy<oy1;oy++){ const unsigned char* r0=&img[(2*oy)*w],*r1=&img[(2*oy+1)*w];
    unsigned int ox=0;
    for(; ox+128<=ow; ox+=128){                                   /* two 128-in chunks -> 128 out px */
      HVX_Vector a0=*(HVX_UVector*)&r0[2*ox],   a1=*(HVX_UVector*)&r1[2*ox];
      HVX_VectorPair sa=Q6_Wuh_vadd_VubVub(a0,a1);                /* lo=even in-pos summed, hi=odd */
      HVX_Vector oa=Q6_Vuh_vlsr_VuhR(Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(Q6_V_lo_W(sa),Q6_V_hi_W(sa)),two),2);
      HVX_Vector b0=*(HVX_UVector*)&r0[2*ox+128],b1=*(HVX_UVector*)&r1[2*ox+128];
      HVX_VectorPair sb=Q6_Wuh_vadd_VubVub(b0,b1);
      HVX_Vector ob=Q6_Vuh_vlsr_VuhR(Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(Q6_V_lo_W(sb),Q6_V_hi_W(sb)),two),2);
      *(HVX_UVector*)&out[oy*ow+ox]=Q6_Vub_vasr_VuhVuhR_rnd_sat(ob,oa,0); /* oa=lo64, ob=hi64, in-order */
    }
    for(; ox<ow; ox++){ unsigned int x=2*ox; out[oy*ow+ox]=(unsigned char)((r0[x]+r0[x+1]+r1[x]+r1[x+1]+2)>>2); }
  }
}
typedef struct { const unsigned char* img; unsigned int w,ow,oy0,oy1; unsigned char* out; worker_synctoken_t* tok; } pyr_band_t;
static void pyr_band_fn(void* pv){ pyr_band_t* b=(pyr_band_t*)pv;
  pyrdown_hvx_rows(b->img,b->w,b->ow,b->out,b->oy0,b->oy1); worker_pool_synctoken_jobdone(b->tok); }

int calculator_pyrdown(remote_handle64 h, const unsigned char* img, int imgLen,
                       unsigned int w, unsigned int hh, unsigned int mode,
                       unsigned char* out, int outLen, unsigned int* us){
  unsigned long long _t0=HAP_perf_get_time_us();
  unsigned int ow=w/2, oh=hh/2;
  if(mode==2){ worker_pool_context_t ctx; worker_pool_init(&ctx);
    unsigned int nw=num_workers; if(nw<1)nw=1; if(nw>16)nw=16; if(nw>oh)nw=oh?oh:1;
    worker_synctoken_t tok; worker_pool_synctoken_init(&tok,nw);
    pyr_band_t bd[16]; unsigned int per=(oh+nw-1)/nw;
    for(unsigned int k=0;k<nw;k++){ unsigned int a=k*per,b=a+per; if(a>oh)a=oh; if(b>oh)b=oh;
      bd[k].img=img;bd[k].w=w;bd[k].ow=ow;bd[k].out=out;bd[k].oy0=a;bd[k].oy1=b;bd[k].tok=&tok;
      worker_pool_job_t j; j.fptr=pyr_band_fn; j.dptr=&bd[k]; worker_pool_submit(ctx,j); }
    worker_pool_synctoken_wait(&tok); worker_pool_deinit(&ctx);
  }
  else if(mode==1) pyrdown_hvx_rows(img,w,ow,out,0,oh);
  else             pyrdown_scalar_rows(img,w,ow,out,0,oh);
  *us=(unsigned int)(HAP_perf_get_time_us()-_t0);
  return 0;
}
