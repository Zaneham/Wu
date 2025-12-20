
import time
import numpy as np
from PIL import Image
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from wu.dimensions.copymove import CopyMoveAnalyzer, HAS_NATIVE_SIMD as HAS_SIMD_CM
from wu.dimensions.blockgrid import BlockGridAnalyzer, HAS_NATIVE_SIMD as HAS_SIMD_BG
from wu.dimensions.lighting import LightingAnalyzer, HAS_NATIVE_SIMD as HAS_SIMD_LI
from wu.dimensions.prnu import PRNUAnalyzer, HAS_NATIVE_SIMD as HAS_SIMD_PRNU
from wu.native import simd

def benchmark_copymove(size=(512, 512)):
    print(f"\n--- CopyMove Benchmark ({size}) ---")
    
    # Create random image
    img_arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(img_arr)
    img.save("bench_cm.jpg")
    
    analyzer = CopyMoveAnalyzer()
    
    # Force Python path (by temporarily sabotaging HAS_NATIVE_SIMD if possible, 
    # but since it's imported at module level, we might need to mock or just rely on the fallback logic inside `copymove.py` 
    # if we could toggle it. But `HAS_NATIVE_SIMD` is a global constant.)
    # Instead, we will toggle the global var in the module if possible, or simpler: 
    # Just measure current performance (which should be Assembly) vs "Slow" path by manually calling logic if needed.
    # Actually, let's just measure "Enabled" performance for now, as that's what matters.
    # Comparing to "Disabled" requires hacking the module.
    
    start_time = time.time()
    result = analyzer.analyze("bench_cm.jpg")
    duration = (time.time() - start_time) * 1000
    
    print(f"Time: {duration:.2f} ms")
    print(f"Method: {result.methodology} (Native: {HAS_SIMD_CM})")
    
    if os.path.exists("bench_cm.jpg"):
        os.remove("bench_cm.jpg")
        
def benchmark_blockgrid(size=(1024, 1024)):
    print(f"\n--- BlockGrid Benchmark ({size}) ---")
    img_arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(img_arr)
    img.save("bench_bg.jpg", quality=95)
    
    analyzer = BlockGridAnalyzer()
    
    start_time = time.time()
    result = analyzer.analyze("bench_bg.jpg")
    duration = (time.time() - start_time) * 1000
    
    print(f"Time: {duration:.2f} ms")
    print(f"Method: Native: {HAS_SIMD_BG}")
    
    if os.path.exists("bench_bg.jpg"):
        os.remove("bench_bg.jpg")

def benchmark_lighting(size=(1024, 1024)):
    print(f"\n--- Lighting Benchmark ({size}) ---")
    img_arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(img_arr)
    # Lighting requires PNG/Lossless usually
    img.save("bench_li.png")
    
    analyzer = LightingAnalyzer()
    
    start_time = time.time()
    result = analyzer.analyze("bench_li.png")
    duration = (time.time() - start_time) * 1000
    
    print(f"Time: {duration:.2f} ms")
    print(f"Method: Native: {HAS_SIMD_LI}")
    
    if os.path.exists("bench_li.png"):
        os.remove("bench_li.png")

def benchmark_prnu(size=(512, 512)):
    print(f"\n--- PRNU PCE Benchmark ({size}) ---")
    # PRNU is heavy.
    noise = np.random.normal(0, 1, size).astype(np.float64)
    fingerprint = np.random.normal(0, 1, size).astype(np.float64)
    
    analyzer = PRNUAnalyzer()
    
    start_time = time.time()
    # Direct cell to _compute_pce
    pce, p_val = analyzer._compute_pce(noise, fingerprint)
    duration = (time.time() - start_time) * 1000
    
    print(f"Time: {duration:.2f} ms")
    print(f"Method: Native: {HAS_SIMD_PRNU}")

if __name__ == "__main__":
    print("Running Benchmarks...")
    simd.print_simd_info()
    
    benchmark_copymove((256, 256))
    benchmark_blockgrid((1024, 1024))
    benchmark_lighting((1024, 1024))
    benchmark_prnu((512, 512))
    
    print("\nBenchmarks Complete.")
