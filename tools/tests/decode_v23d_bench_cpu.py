"""CPU (TRITON_INTERPRET) check of decode v23d_bench.py's diagnostic kernels: _gather_probe sums == torch sums; _attn_rows_pref (PREF / HOIST on and off) == decode v19's _attn_split_rows bit for bit."""
import os,sys,types,importlib.util
os.environ["TRITON_INTERPRET"]="1"
sys.argv=[sys.argv[0]]
sys.path.insert(0,"/workspace/okc-decode-v23/tools/tests")
# reuse stubs from glue test by exec of its top part
src=open("/workspace/okc-decode-v23/tools/tests/decode_v23_fv_glue.py").read().split("spec = importlib.util.spec_from_file_location(\"dec\", path)")[0]
src=src.replace('path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "muse_decode_layer", "runs", "v23_fv_attn.py")','path="/workspace/okc-decode-v23/muse_decode_layer/runs/v23d_bench.py"')
exec(src)
spec=importlib.util.spec_from_file_location("d",path); D=importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
import torch
torch.manual_seed(0)
batch,cap,W,hq,d=2,300,128,16,64
total=batch*cap
pt=torch.randperm(total).to(torch.int32).view(batch,cap).contiguous()
sl=torch.tensor([250,299],dtype=torch.int32)
kc=(torch.randint(0,256,(total,W),dtype=torch.int32)&0x77).to(torch.uint8)
vc=(torch.randint(0,256,(total,W),dtype=torch.int32)&0x77).to(torch.uint8)
q=(torch.randn(batch,hq*d)).float()
one=torch.ones(())
for nsp,bn,pref,contig in [(4,64,False,False),(4,64,True,False),(2,128,False,True)]:
    chunk=-(-cap//nsp); out=torch.zeros(batch*nsp,dtype=torch.int32)
    D._gather_probe[(nsp,batch)](kc,vc,pt,pt.stride(0),sl,out,chunk,NSPLIT=nsp,BLOCK_N=bn,W=W,PREF=pref,CONTIG=contig)
    exp=0
    for b in range(batch):
        rows=(torch.arange(int(sl[b]))+b*cap) if contig else pt[b,:int(sl[b])].long()
        exp+=int(kc[rows].int().sum()+vc[rows].int().sum())
    print("probe",nsp,bn,pref,contig,int(out.sum())==exp)
res={}
for name,fn,kw in [("v19",D._attn_split_rows,dict(HOIST=True)),("pref",D._attn_rows_pref,dict(HOIST=True,PREF=True)),("nopref",D._attn_rows_pref,dict(HOIST=False,PREF=False))]:
    nsp=4; chunk=-(-cap//nsp)
    o=torch.zeros(batch*hq*nsp,d); l=torch.zeros(batch*hq*nsp)
    fn[(nsp,batch)](q,q.stride(0),kc,vc,pt,pt.stride(0),sl,one,one,o,l,0.125,chunk,1,L=1,G=8,HKV=2,NSPLIT=nsp,WINDOW=0,BLOCK_M=16,BLOCK_N=64,D=d,HW=False,PV16=False,**kw)
    res[name]=(o,l)
for k in ("pref","nopref"): print(k, (res[k][0]-res["v19"][0]).abs().max().item(), (res[k][1]-res["v19"][1]).abs().max().item())
