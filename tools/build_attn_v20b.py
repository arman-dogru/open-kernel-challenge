"""Build muse_attention_decode/runs/v20b_verify_pref64.py: tools/build_attn_v20.py with sliding verify on v17's 64-key
head tiles plus V hoisted and next-tile page-index prefetch (HOIST + PREF), i.e. ("head", 64, 4, 3, 4, True).
v20 check 77bcf164955a4ca3 (1.394x): 128-key tiles made the split kernel slower (verify 16 10.0 -> 11.7 us, verify 32
16.2 -> 22.8), so the tile size stays 64 and only the load overlap is kept. Judge dtypes: 214 regs, 0 stack (v17 234).
"""
import os, runpy, sys
os.environ["V20_VERIFY_CFG"] = '("head", 64, 4, 3, 4, True)'
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "muse_attention_decode/runs/v20b_verify_pref64.py"]
runpy.run_path(os.path.join(os.path.dirname(__file__), "build_attn_v20.py"), run_name="__main__")
