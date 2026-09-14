/* FAST-9-16 corner detector for the Qualcomm Hexagon cDSP (v73/HVX).
 * Original work (VSLAM-IQ9). Designed to be invoked as a fastRPC skel function
 * (mode=0 scalar reference, mode=1 HVX). Integrates with the Hexagon SDK
 * fastRPC 'calculator' example scaffold, which is NOT included (proprietary;
 * obtain the Hexagon SDK separately). Only this kernel is our original code.
 *
 * HVX approach for the 9-contiguous-of-16 arc test: vectorized run-length over
 * k=0..23 (wrapped) across 128 u8 lanes; corner if max consecutive brighter
 * (or darker) run >= 9. Right-edge remainder via a masked final HVX vector.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include "HAP_perf.h"

int calculator_detect(remote_handle64 h, const unsigned char* img, int imgLen,
                      unsigned int w, unsigned int hh, unsigned int barrier, unsigned int mode,
                      unsigned int* xy, int xyLen, unsigned int* ncorn, unsigned int* us) {
  unsigned long long _t0=HAP_perf_get_time_us();
  if(mode==1){
    static const int ox2[16]={0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1};
    static const int oy2[16]={-3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3};
    const int VLEN=128; unsigned int total=0;
    HVX_Vector vt=Q6_Vb_vsplat_R((int)barrier);
    HVX_Vector one=Q6_Vb_vsplat_R(1), z=Q6_V_vzero(), eight=Q6_Vb_vsplat_R(8);
    for(unsigned int y=3;y<hh-3;y++){
      HVX_Vector accw=Q6_V_vzero(); unsigned int xlast=3;
      for(unsigned int x=3;x+VLEN<=w-3;x+=VLEN){
        const unsigned char* c=&img[y*w+x];
        HVX_Vector C=*(HVX_UVector*)c;
        HVX_Vector hi=Q6_Vub_vadd_VubVub_sat(C,vt);
        HVX_Vector lo=Q6_Vub_vsub_VubVub_sat(C,vt);
        HVX_Vector runB=z,runD=z,maxB=z,maxD=z;
        for(int j=0;j<24;j++){ int k=j&15;
          HVX_Vector Vk=*(HVX_UVector*)&img[(y+oy2[k])*w + x+ox2[k]];
          HVX_VectorPred pB=Q6_Q_vcmp_gt_VubVub(Vk,hi);
          HVX_VectorPred pD=Q6_Q_vcmp_gt_VubVub(lo,Vk);
          runB=Q6_V_vmux_QVV(pB,Q6_Vb_vadd_VbVb(runB,one),z);
          runD=Q6_V_vmux_QVV(pD,Q6_Vb_vadd_VbVb(runD,one),z);
          maxB=Q6_Vub_vmax_VubVub(maxB,runB);
          maxD=Q6_Vub_vmax_VubVub(maxD,runD);
        }
        HVX_VectorPred cB=Q6_Q_vcmp_gt_VubVub(maxB,eight);
        HVX_VectorPred cD=Q6_Q_vcmp_gt_VubVub(maxD,eight);
        HVX_Vector corner=Q6_V_vmux_QVV(cB,one,Q6_V_vmux_QVV(cD,one,z)); /* 1 per corner lane */
        accw=Q6_Vuw_vrmpyacc_VuwVubRub(accw,corner,0x01010101u); /* sum bytes -> words */
        xlast=x+VLEN;
      }
      /* horizontal-sum the 32 word lanes of accw */
      unsigned int tmp[32]; *(HVX_UVector*)tmp=accw; for(int q=0;q<32;q++) total+=tmp[q];
      /* HVX tail: one final vector at the right edge, mask off already-counted lanes */
      if(xlast < w-3){
        unsigned int xf=w-3-VLEN, M=xlast-xf; /* count lanes [M,128) => global x in [xlast, w-3) */
        unsigned char mb[128]; for(int q=0;q<128;q++) mb[q]=(q>=(int)M)?1:0;
        HVX_Vector mask=*(HVX_UVector*)mb;
        const unsigned char* c=&img[y*w+xf];
        HVX_Vector C=*(HVX_UVector*)c;
        HVX_Vector hi=Q6_Vub_vadd_VubVub_sat(C,vt), lo=Q6_Vub_vsub_VubVub_sat(C,vt);
        HVX_Vector runB=z,runD=z,maxB=z,maxD=z;
        for(int j=0;j<24;j++){ int k=j&15;
          HVX_Vector Vk=*(HVX_UVector*)&img[(y+oy2[k])*w + xf+ox2[k]];
          HVX_VectorPred pB=Q6_Q_vcmp_gt_VubVub(Vk,hi), pD=Q6_Q_vcmp_gt_VubVub(lo,Vk);
          runB=Q6_V_vmux_QVV(pB,Q6_Vb_vadd_VbVb(runB,one),z);
          runD=Q6_V_vmux_QVV(pD,Q6_Vb_vadd_VbVb(runD,one),z);
          maxB=Q6_Vub_vmax_VubVub(maxB,runB); maxD=Q6_Vub_vmax_VubVub(maxD,runD);
        }
        HVX_VectorPred cB=Q6_Q_vcmp_gt_VubVub(maxB,eight), cD=Q6_Q_vcmp_gt_VubVub(maxD,eight);
        HVX_Vector corner=Q6_V_vmux_QVV(cB,one,Q6_V_vmux_QVV(cD,one,z));
        corner=Q6_V_vand_VV(corner,mask);
        HVX_Vector a2=Q6_Vuw_vrmpy_VubRub(corner,0x01010101u);
        unsigned int t2[32]; *(HVX_UVector*)t2=a2; for(int q=0;q<32;q++) total+=t2[q];
      }
    }
    *ncorn=total; *us=(unsigned int)(HAP_perf_get_time_us()-_t0);
    FARF(RUNTIME_HIGH,"DSP HVX FAST: %u corners %u us",total,*us);
    return 0;
  }

  static const int ox[16]={0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1};
  static const int oy[16]={-3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3};
  unsigned int cnt=0; int maxc=xyLen/2;
  for(unsigned int y=3;y<hh-3;y++){
    for(unsigned int x=3;x<w-3;x++){
      int p=img[y*w+x], hi=p+(int)barrier, lo=p-(int)barrier; int br=0,dk=0,k;
      for(k=0;k<16;k++){int v=img[(y+oy[k])*w+(x+ox[k])]; br|=(v>hi)<<k; dk|=(v<lo)<<k;}
      int b32=br|(br<<16), d32=dk|(dk<<16), isc=0,s;
      for(s=0;s<16;s++){ if(((b32>>s)&0x1FF)==0x1FF || ((d32>>s)&0x1FF)==0x1FF){isc=1;break;} }
      if(isc){ if(cnt<(unsigned int)maxc){xy[2*cnt]=x;xy[2*cnt+1]=y;} cnt++; }
    }
  }
  *ncorn=cnt; *us=(unsigned int)(HAP_perf_get_time_us()-_t0); FARF(RUNTIME_HIGH,"DSP FAST: %u corners %u us",cnt,*us); return 0;
}
