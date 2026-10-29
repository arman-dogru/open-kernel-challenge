"""Offline sm_90 type-check + compile of every Triton launch an attention submission makes (no GPU needed).

  capture: python triton_sm90_compile.py capture <submission.py> <launches.pkl>   (TRITON_INTERPRET=1, CPU torch;
           runs the CPU cases below and records each launch's kernel, argument types and constexprs)
  compile: python triton_sm90_compile.py compile <submission.py> <launches.pkl>   (no interpreter; triton.compile of each
           unique specialization for GPUTarget("cuda", 90, 128), then cuobjdump for registers / stack)
  both:    python triton_sm90_compile.py <submission.py>                          (runs both in subprocesses)

Run from muse_attention_decode/. The interpreter does not type-check (attention v18's check 639cbea473a548e1 failed on an
if-branch redefining a variable's type); the compile step runs the real code generator, so that class of error fails here.
"""
import os, sys, pickle, inspect, subprocess, importlib.util, re, tempfile

TORCH2TT = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16", "float8_e4m3fn": "fp8e4nv", "float8_e5m2": "fp8e5",
            "int32": "i32", "int64": "i64", "int16": "i16", "int8": "i8", "uint8": "u8", "bool": "i1"}
CASES = [dict(batch=8, ctx=2600, layer="sliding", q_len=8, spread="agentic"), dict(batch=8, ctx=2600, layer="sliding", q_len=8, kv_scale=0.5), dict(batch=8, ctx=900, layer="full", q_len=8, page=64)] + [dict(batch=1, ctx=3000, layer="sliding", q_len=1), dict(batch=3, ctx=2500, layer="sliding", q_len=1, spread="agentic"),
         dict(batch=2, ctx=2600, layer="sliding", q_len=8), dict(batch=3, ctx=2600, layer="sliding", q_len=8, spread="agentic"),
         dict(batch=1, ctx=2600, layer="sliding", q_len=8),
         dict(batch=1, ctx=900, layer="full", q_len=8, dist="coherent"), dict(batch=2, ctx=900, layer="full", q_len=8, page=64),
         dict(batch=2, ctx=700, layer="full", q_len=1), dict(batch=1, ctx=700, layer="full", q_len=1)]


def load(p):
    spec = importlib.util.spec_from_file_location(os.path.basename(p)[:-3], p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def capture(sub, out):
    os.environ["TRITON_INTERPRET"] = "1"
    sys.path.insert(0, os.getcwd())
    import torch
    from triton.runtime.interpreter import InterpretedFunction
    import reference as R
    seen = {}
    orig = InterpretedFunction.run

    def run(self, *args, grid, warmup, **kwargs):
        sig = inspect.signature(self.fn)
        opts = {k: kwargs.pop(k) for k in ("num_warps", "num_stages") if k in kwargs}
        bound = sig.bind_partial(*args, **kwargs); bound.apply_defaults()
        rec = []
        for name, p in sig.parameters.items():
            v = bound.arguments.get(name)
            is_c = "constexpr" in str(p.annotation)
            if is_c or v is None:
                rec.append((name, "constexpr", v if not hasattr(v, "value") else v.value))
            elif isinstance(v, torch.Tensor):
                rec.append((name, "*" + TORCH2TT[str(v.dtype).split(".")[-1]], None))
            elif isinstance(v, bool):
                rec.append((name, "i1", None))
            elif isinstance(v, int):
                rec.append((name, "i64" if abs(v) >= 2**31 else "i32", None))
            elif isinstance(v, float):
                rec.append((name, "fp32", None))
            else:
                raise TypeError(f"{self.fn.__name__}.{name}: {type(v)}")
        key = (self.fn.__name__, tuple(rec), tuple(sorted(opts.items())))
        seen[key] = seen.get(key, 0) + 1
        return orig(self, *args, grid=grid, warmup=warmup, **kwargs, **opts)

    InterpretedFunction.run = run
    m = load(sub)
    for c in CASES:
        d = R.generate_input(heads="tp8", seed=3, device="cpu", **c)
        m.kernel_fn(d)
    pickle.dump(list(seen), open(out, "wb"))
    print(f"captured {len(seen)} unique launches", flush=True)


def compile_all(sub, inp):
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    m = load(sub)
    cuobjdump = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")
    bad = 0
    for name, rec, opts in pickle.load(open(inp, "rb")):
        fn = getattr(m, name)
        sig = {n: t for n, t, _ in rec}
        cx = {n: v for n, t, v in rec if t == "constexpr"}
        o = dict(opts)
        short = {k: v for k, v in cx.items() if isinstance(v, (int, bool, str))}
        try:
            k = triton.compile(ASTSource(fn, sig, cx), target=GPUTarget("cuda", 90, 128), options=o)
            with tempfile.TemporaryDirectory() as td:  # resource usage of the cubin triton.compile produced
                cb = os.path.join(td, "k.cubin"); open(cb, "wb").write(k.asm["cubin"])
                r = subprocess.run([cuobjdump, "--dump-resource-usage", cb], capture_output=True, text=True)
            regs = re.findall(r"REG:(\d+)", r.stdout); sp = re.findall(r"STACK:(\d+)", r.stdout)
            print(f"OK   {name} {o} regs={regs} stack={sp} {short}", flush=True)
        except Exception as e:
            bad += 1
            print(f"FAIL {name} {o} {short}\n     {type(e).__name__}: {str(e)[-600:]}", flush=True)
    print("compile failures:", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "compile":
        compile_all(sys.argv[2], sys.argv[3])
    else:
        with tempfile.TemporaryDirectory() as td:
            pk = os.path.join(td, "l.pkl")
            env = dict(os.environ); env.pop("TRITON_INTERPRET", None)
            subprocess.run([sys.executable, __file__, "capture", sys.argv[1], pk], check=True)
            sys.exit(subprocess.run([sys.executable, __file__, "compile", sys.argv[1], pk], env=env).returncode)
