import ctypes as C, numpy as np
cl=C.CDLL("libOpenCL.so.1")
u=C.c_uint32; i=C.c_int32; vp=C.c_void_p; sz=C.c_size_t
cl.clGetPlatformIDs.argtypes=[u,C.POINTER(vp),C.POINTER(u)]
cl.clGetDeviceIDs.argtypes=[vp,C.c_uint64,u,C.POINTER(vp),C.POINTER(u)]
cl.clCreateContext.restype=vp; cl.clCreateContext.argtypes=[vp,u,C.POINTER(vp),vp,vp,C.POINTER(i)]
cl.clCreateCommandQueue.restype=vp; cl.clCreateCommandQueue.argtypes=[vp,vp,C.c_uint64,C.POINTER(i)]
cl.clCreateBuffer.restype=vp; cl.clCreateBuffer.argtypes=[vp,C.c_uint64,sz,vp,C.POINTER(i)]
cl.clCreateProgramWithSource.restype=vp; cl.clCreateProgramWithSource.argtypes=[vp,u,C.POINTER(C.c_char_p),C.POINTER(sz),C.POINTER(i)]
cl.clBuildProgram.argtypes=[vp,u,C.POINTER(vp),C.c_char_p,vp,vp]
cl.clCreateKernel.restype=vp; cl.clCreateKernel.argtypes=[vp,C.c_char_p,C.POINTER(i)]
cl.clSetKernelArg.argtypes=[vp,u,sz,vp]
cl.clEnqueueWriteBuffer.argtypes=[vp,vp,u,sz,sz,vp,u,vp,vp]
cl.clEnqueueNDRangeKernel.argtypes=[vp,vp,u,vp,C.POINTER(sz),vp,u,vp,vp]
cl.clEnqueueReadBuffer.argtypes=[vp,vp,u,sz,sz,vp,u,vp,vp]
cl.clFinish.argtypes=[vp]
plat=vp(); n=u(); cl.clGetPlatformIDs(1,C.byref(plat),C.byref(n))
dev=vp(); cl.clGetDeviceIDs(plat,1<<2,1,C.byref(dev),C.byref(n))  # CL_DEVICE_TYPE_GPU=1<<2
err=i(); ctx=cl.clCreateContext(None,1,C.byref(dev),None,None,C.byref(err)); assert err.value==0,("ctx",err.value)
q=cl.clCreateCommandQueue(ctx,dev,0,C.byref(err)); assert err.value==0,("q",err.value)
N=1<<20; a=np.arange(N,dtype=np.float32); out=np.zeros(N,np.float32)
src=b"__kernel void k(__global const float*a,__global float*o){int g=get_global_id(0); o[g]=a[g]*2.0f+1.0f;}"
srcp=C.c_char_p(src); L=sz(len(src))
prog=cl.clCreateProgramWithSource(ctx,1,C.byref(srcp),C.byref(L),C.byref(err)); assert err.value==0,("prog",err.value)
be=cl.clBuildProgram(prog,1,C.byref(dev),None,None,None); assert be==0,("build",be)
k=cl.clCreateKernel(prog,b"k",C.byref(err)); assert err.value==0,("kern",err.value)
CL_MEM_RW=1; CL_MEM_COPY=1<<5
ba=cl.clCreateBuffer(ctx,CL_MEM_RW|CL_MEM_COPY,4*N,a.ctypes.data_as(vp),C.byref(err))
bo=cl.clCreateBuffer(ctx,CL_MEM_RW,4*N,None,C.byref(err))
cl.clSetKernelArg(k,0,sz(C.sizeof(vp)),C.byref(C.c_void_p(ba)))
cl.clSetKernelArg(k,1,sz(C.sizeof(vp)),C.byref(C.c_void_p(bo)))
gws=(sz*1)(N); cl.clEnqueueNDRangeKernel(q,k,1,None,gws,None,0,None,None); cl.clFinish(q)
cl.clEnqueueReadBuffer(q,bo,1,0,4*N,out.ctypes.data_as(vp),0,None,None); cl.clFinish(q)
exp=a*2+1
print("FD663 kernel exec OK:",bool(np.allclose(out,exp)),"| sample",out[:3],"expect",exp[:3])
