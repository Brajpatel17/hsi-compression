"""Runs the whole study in order: split -> classical -> neural -> figures.
Total time on a laptop CPU: roughly 45-90 minutes (the 3D-CNN is the slow part)."""
import os, subprocess, sys, time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
steps = ["split_data.py",
         "classical/run_classical.py",
         "neural/run_mlp.py", "neural/run_cnn1d.py", "neural/run_cnn3d.py",
         "plot_results.py"]
for s in steps:
    t = time.time()
    print(f"\n===== {s} =====", flush=True)
    subprocess.run([sys.executable, s], check=True)
    print(f"----- {s} done in {time.time() - t:.0f}s", flush=True)
