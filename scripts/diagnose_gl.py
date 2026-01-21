#!/usr/bin/env python3
"""
Diagnostic script to check OpenGL/OSMesa setup on cluster.
Run this BEFORE running eval_baseline.py to diagnose rendering issues.
"""

import os
import sys
import subprocess

def run_cmd(cmd):
    """Run shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip() + result.stderr.strip()
    except Exception as e:
        return str(e)

def main():
    print("=" * 60)
    print("OpenGL/OSMesa Diagnostic Report")
    print("=" * 60)
    
    # Environment variables
    print("\n1. Environment Variables:")
    print("-" * 40)
    for var in ["MUJOCO_GL", "PYOPENGL_PLATFORM", "DISPLAY", "LD_LIBRARY_PATH", "LIBGL_ALWAYS_SOFTWARE"]:
        print(f"  {var}: {os.environ.get(var, '(not set)')}")
    
    # Check for OSMesa libraries
    print("\n2. OSMesa Library Check:")
    print("-" * 40)
    
    print("  ldconfig -p | grep -i osmesa:")
    print(f"    {run_cmd('ldconfig -p | grep -i osmesa') or '(none found)'}")
    
    print("  Looking for libOSMesa*.so:")
    for path in ["/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu"]:
        result = run_cmd(f"ls {path}/libOSMesa*.so 2>/dev/null")
        if result:
            print(f"    Found in {path}: {result}")
    
    # Check for header
    print("\n  OSMesa header:")
    header_check = run_cmd("ls /usr/include/GL/osmesa.h 2>/dev/null")
    print(f"    {header_check or '(not found)'}")
    
    # Python OpenGL info
    print("\n3. Python OpenGL Info:")
    print("-" * 40)
    try:
        import OpenGL
        print(f"  PyOpenGL version: {OpenGL.__version__}")
        print(f"  PyOpenGL location: {OpenGL.__file__}")
    except ImportError:
        print("  PyOpenGL not installed!")
    
    # Test osmesa context creation
    print("\n4. OSMesa Context Test:")
    print("-" * 40)
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    try:
        from OpenGL import osmesa
        print("  osmesa module imported successfully")
        
        # Try to create context
        ctx = osmesa.OSMesaCreateContext(osmesa.OSMESA_RGBA, None)
        if ctx:
            print("  ✓ OSMesa context created successfully!")
            osmesa.OSMesaDestroyContext(ctx)
        else:
            print("  ✗ OSMesa context creation returned None")
    except Exception as e:
        print(f"  ✗ OSMesa test failed: {e}")
    
    # Test EGL as fallback
    print("\n5. EGL Backend Test:")
    print("-" * 40)
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    try:
        from OpenGL import EGL
        print("  EGL module imported successfully")
        display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
        if display != EGL.EGL_NO_DISPLAY:
            print("  ✓ EGL display obtained")
        else:
            print("  ✗ EGL display is EGL_NO_DISPLAY")
    except Exception as e:
        print(f"  ✗ EGL test failed: {e}")
    
    # MuJoCo info
    print("\n6. MuJoCo Info:")
    print("-" * 40)
    try:
        import mujoco
        print(f"  MuJoCo version: {mujoco.__version__}")
        # Try headless rendering
        print("  Testing MuJoCo renderer...")
        os.environ["MUJOCO_GL"] = "osmesa"
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><light/><geom type='sphere' size='1'/></worldbody></mujoco>")
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, 64, 64)
        renderer.update_scene(data)
        img = renderer.render()
        print(f"  ✓ MuJoCo render successful! Image shape: {img.shape}")
    except Exception as e:
        print(f"  ✗ MuJoCo test failed: {e}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("Recommendations:")
    print("=" * 60)
    
    if not run_cmd('ldconfig -p | grep -i osmesa'):
        print("""
  OSMesa library NOT FOUND on this system.
  
  Options:
  1. Ask cluster admin to install: sudo apt install libosmesa6 libosmesa6-dev
  2. Try Xvfb: xvfb-run -a python eval_baseline.py ...
  3. Build Mesa with OSMesa locally (see CLUSTER_ISSUES.md)
""")
    else:
        print("""
  OSMesa library found but context creation may still fail.
  
  Try:
  1. export LIBGL_ALWAYS_SOFTWARE=1
  2. export MESA_GL_VERSION_OVERRIDE=3.3
  3. Ensure LD_LIBRARY_PATH includes the OSMesa lib directory
""")

if __name__ == "__main__":
    main()
