// One register-resident row: channel scaling, H128, BF16 rounding, token INT8.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

template<int K,int Threads> __global__ void rotated_quant(const __nv_bfloat16* x,
    const float* divisor,int8_t* q,float* scales) {
    int row=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32;
    __shared__ float partial[32];
    float values[K/Threads];
    float maximum=0.f;
#pragma unroll
    for(int i=0;i<K/(Threads*4);++i) {
        float v[4];
#pragma unroll
        for(int j=0;j<4;++j) {
            int col=threadIdx.x*4+i*(Threads*4)+j;
            v[j]=__bfloat162float(x[row*K+col])/divisor[col];
        }
        float a=v[0]+v[1],b=v[0]-v[1],c=v[2]+v[3],d=v[2]-v[3];
        v[0]=a+c;v[1]=b+d;v[2]=a-c;v[3]=b-d;
#pragma unroll
        for(int delta=1;delta<32;delta*=2) {
#pragma unroll
            for(int j=0;j<4;++j) {
                float other=__shfl_xor_sync(0xffffffff,v[j],delta);
                v[j]=(lane&delta)?other-v[j]:v[j]+other;
            }
        }
#pragma unroll
        for(int j=0;j<4;++j) {
            float result=__bfloat162float(__float2bfloat16_rn(v[j]*0.08838834764831844f));
            values[i*4+j]=result;
            maximum=fmaxf(maximum,fabsf(result));
        }
    }
    for(int delta=16;delta;delta>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,delta));
    if(lane==0)partial[warp]=maximum;
    __syncthreads();
    if(warp==0) {
        maximum=lane<Threads/32?partial[lane]:0.f;
        for(int delta=16;delta;delta>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,delta));
        if(lane==0)partial[0]=maximum>0.f?maximum/127.f:1.f;
    }
    __syncthreads();
    float scale=partial[0];
    if(threadIdx.x==0)scales[row]=scale;
#pragma unroll
    for(int i=0;i<K/(Threads*4);++i) {
        char4 packed;
        packed.x=static_cast<int8_t>(max(-127,min(127,__float2int_rn(values[i*4]/scale))));
        packed.y=static_cast<int8_t>(max(-127,min(127,__float2int_rn(values[i*4+1]/scale))));
        packed.z=static_cast<int8_t>(max(-127,min(127,__float2int_rn(values[i*4+2]/scale))));
        packed.w=static_cast<int8_t>(max(-127,min(127,__float2int_rn(values[i*4+3]/scale))));
        reinterpret_cast<char4*>(q+row*K)[threadIdx.x+i*Threads]=packed;
    }
}
extern "C" int ei_rotated_int8(const void* x,const void* divisor,void* q,
    void* scales,int m,int k,void* stream) {
    if(m<=0)return -1;
    auto s=static_cast<cudaStream_t>(stream);
    if(k==2048)rotated_quant<2048,256><<<m,256,0,s>>>(static_cast<const __nv_bfloat16*>(x),static_cast<const float*>(divisor),static_cast<int8_t*>(q),static_cast<float*>(scales));
    else if(k==16384)rotated_quant<16384,512><<<m,512,0,s>>>(static_cast<const __nv_bfloat16*>(x),static_cast<const float*>(divisor),static_cast<int8_t*>(q),static_cast<float*>(scales));
    else return -1;
    return cudaGetLastError();
}
