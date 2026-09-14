import ctypes as C, numpy as np, time, cv2
cl=C.CDLL("libOpenCL.so.1")
u=C.c_uint32;i=C.c_int32;vp=C.c_void_p;sz=C.c_size_t
for f,r,a in [("clGetPlatformIDs",i,[u,C.POINTER(vp),C.POINTER(u)]),("clGetDeviceIDs",i,[vp,C.c_uint64,u,C.POINTER(vp),C.POINTER(u)]),("clCreateContext",vp,[vp,u,C.POINTER(vp),vp,vp,C.POINTER(i)]),("clCreateCommandQueue",vp,[vp,vp,C.c_uint64,C.POINTER(i)]),("clCreateBuffer",vp,[vp,C.c_uint64,sz,vp,C.POINTER(i)]),("clCreateProgramWithSource",vp,[vp,u,C.POINTER(C.c_char_p),C.POINTER(sz),C.POINTER(i)]),("clBuildProgram",i,[vp,u,C.POINTER(vp),C.c_char_p,vp,vp]),("clCreateKernel",vp,[vp,C.c_char_p,C.POINTER(i)]),("clSetKernelArg",i,[vp,u,sz,vp]),("clEnqueueWriteBuffer",i,[vp,vp,u,sz,sz,vp,u,vp,vp]),("clEnqueueNDRangeKernel",i,[vp,vp,u,vp,C.POINTER(sz),vp,u,vp,vp]),("clEnqueueReadBuffer",i,[vp,vp,u,sz,sz,vp,u,vp,vp]),("clFinish",i,[vp])]:
    fn=getattr(cl,f);fn.restype=r;fn.argtypes=a
plat=vp();n=u();cl.clGetPlatformIDs(1,C.byref(plat),C.byref(n))
dev=vp();cl.clGetDeviceIDs(plat,1<<2,1,C.byref(dev),C.byref(n))
err=i();ctx=cl.clCreateContext(None,1,C.byref(dev),None,None,C.byref(err))
q=cl.clCreateCommandQueue(ctx,dev,0,C.byref(err))
img=cv2.imread("/opt/orbslam/data/MH_01/mav0/cam0/data/1403636579763555584.png",0).astype(np.float32)
H,W=img.shape;flat=np.ascontiguousarray(img.ravel());out=np.zeros_like(flat)
src=b"__kernel void box3(__global const float*in,__global float*o,int W,int H){int x=get_global_id(0),y=get_global_id(1); if(x>=W||y>=H)return; float s=0.0f; for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){int xx=clamp(x+dx,0,W-1),yy=clamp(y+dy,0,H-1);s+=in[yy*W+xx];} o[y*W+x]=s/9.0f;}"
sp=C.c_char_p(src);L=sz(len(src));prog=cl.clCreateProgramWithSource(ctx,1,C.byref(sp),C.byref(L),C.byref(err))
be=cl.clBuildProgram(prog,1,C.byref(dev),None,None,None);assert be==0,("build",be)
k=cl.clCreateKernel(prog,b"box3",C.byref(err))
bi=cl.clCreateBuffer(ctx,1|(1<<5),4*flat.size,flat.ctypes.data_as(vp),C.byref(err))
bo=cl.clCreateBuffer(ctx,1,4*flat.size,None,C.byref(err))
cl.clSetKernelArg(k,0,sz(8),C.byref(C.c_void_p(bi)));cl.clSetKernelArg(k,1,sz(8),C.byref(C.c_void_p(bo)))
cl.clSetKernelArg(k,2,sz(4),C.byref(i(W)));cl.clSetKernelArg(k,3,sz(4),C.byref(i(H)))
gws=(sz*2)(W,H)
def gpu():
    cl.clEnqueueNDRangeKernel(q,k,2,None,gws,None,0,None,None);cl.clFinish(q)
def gpu_e2e():
    cl.clEnqueueWriteBuffer(q,bi,1,0,4*flat.size,flat.ctypes.data_as(vp),0,None,None)
    cl.clEnqueueNDRangeKernel(q,k,2,None,gws,None,0,None,None)
    cl.clEnqueueReadBuffer(q,bo,1,0,4*flat.size,out.ctypes.data_as(vp),0,None,None);cl.clFinish(q)
def med(fn,N=60,w=12):
    t=[]
    for _ in range(N):
        s=time.perf_counter();fn();t.append((time.perf_counter()-s)*1e3)
    return float(np.median(t[w:]))
mg=med(gpu);me=med(gpu_e2e)
cl.clEnqueueReadBuffer(q,bo,1,0,4*flat.size,out.ctypes.data_as(vp),0,None,None);cl.clFinish(q)
ref=cv2.blur(img,(3,3),borderType=cv2.BORDER_REPLICATE)
mad=float(np.max(np.abs(out.reshape(H,W)-ref)))
def cpu():cv2.blur(img,(3,3),borderType=cv2.BORDER_REPLICATE)
mc=med(cpu)
verdict="PASS" if mad<1e-2 else "FAIL"
print("Adreno FD663 3x3 blur 480x752: kernel-only %.3f ms | e2e(HtoD+DtoH) %.3f ms"%(mg,me))
print("CPU cv2.blur: %.3f ms"%mc)
print("GATE max|GPU-CPUref| = %.5f -> %s"%(mad,verdict))
print("speedup kernel-only %.2fx | e2e %.2fx"%(mc/mg,mc/me))
