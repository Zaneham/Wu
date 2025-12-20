import sys
import os
import unittest
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Debug print
print(f"Sys Path: {sys.path}")

try:
    from wu.dimensions.copymove import CopyMoveAnalyzer, HAS_NATIVE_SIMD as HAS_SIMD_CM
    from wu.dimensions.blockgrid import BlockGridAnalyzer, HAS_NATIVE_SIMD as HAS_SIMD_BG
    from wu.dimensions.lighting import LightingAnalyzer, HAS_NATIVE_SIMD as HAS_SIMD_LI
except ImportError as e:
    print(f"Import Error during module load: {e}")
    HAS_SIMD_CM = False
    HAS_SIMD_BG = False
    HAS_SIMD_LI = False

class TestDimensionsAsm(unittest.TestCase):
    def setUp(self):
        # Create dummy image
        self.width = 256
        self.height = 256
        self.image_array = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        self.image = Image.fromarray(self.image_array)
        
        # Save as JPEG for blockgrid
        self.jpeg_path = "test_image.jpg"
        self.image.save(self.jpeg_path, quality=95)
        
        # Save as PNG for others
        self.png_path = "test_image.png"
        self.image.save(self.png_path)

    def tearDown(self):
        if os.path.exists(self.jpeg_path):
            os.remove(self.jpeg_path)
        if os.path.exists(self.png_path):
            os.remove(self.png_path)
            
    def test_debug_simd_import(self):
        print("\n--- DEBUG SIMD IMPORT ---")
        try:
            import wu.native.simd as s
            print(f"Direct import wu.native.simd: {s}")
            print(f"s.is_available(): {s.is_available()}")
            print(f"s._lib: {s._lib}")
        except Exception as e:
            print(f"Direct import failed: {e}")
            import traceback
            traceback.print_exc()

    def test_native_simd_available(self):
        print("\nChecking Native SIMD Availability in Modules:")
        print(f"CopyMove.HAS_NATIVE_SIMD: {HAS_SIMD_CM}")
        print(f"BlockGrid.HAS_NATIVE_SIMD: {HAS_SIMD_BG}")
        print(f"Lighting.HAS_NATIVE_SIMD: {HAS_SIMD_LI}")
        
        # We assert they are TRUE. If failing, we want to know why.
        self.assertTrue(HAS_SIMD_CM, "CopyMove SIMD not available")
        self.assertTrue(HAS_SIMD_BG, "BlockGrid SIMD not available")
        self.assertTrue(HAS_SIMD_LI, "Lighting SIMD not available")

    def test_copymove_analyze(self):
        print("\nTesting CopyMove Analysis...")
        if not HAS_SIMD_CM:
            print("WARNING: Skipping assembly check for CopyMove (unavailable)")
            return
            
        try:
            analyzer = CopyMoveAnalyzer()
            analyzer.SIMILARITY_THRESHOLD = 0.8 
            analyzer.MIN_CLONE_DISTANCE = 10
            
            clone_src = self.image_array[0:32, 0:32].copy()
            temp_arr = self.image_array.copy()
            temp_arr[100:132, 100:132] = clone_src
            img_clone = Image.fromarray(temp_arr)
            img_clone.save("test_clone.png")
            
            result = analyzer.analyze("test_clone.png")
            print(f"CopyMove Result: {result.state}")
            
            if os.path.exists("test_clone.png"):
                os.remove("test_clone.png")

        except Exception as e:
            print(f"TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_blockgrid_analyze(self):
        print("\nTesting BlockGrid Analysis...")
        if not HAS_SIMD_BG:
            print("WARNING: Skipping assembly check for BlockGrid (unavailable)")
            return
            
        analyzer = BlockGridAnalyzer()
        result = analyzer.analyze(self.jpeg_path)
        print(f"BlockGrid Result: {result.state}")

    def test_lighting_analyze(self):
        print("\nTesting Lighting Analysis...")
        if not HAS_SIMD_LI:
            print("WARNING: Skipping assembly check for Lighting (unavailable)")
            return
            
        analyzer = LightingAnalyzer()
        result = analyzer.analyze(self.png_path)
        print(f"Lighting Result: {result.state}")

if __name__ == '__main__':
    unittest.main()
