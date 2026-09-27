"""Run every make_figure*.py script in turn (from the archived fits/*.json) and
report which succeeded. Usage: python make_all_figures.py [1 2 5 A1 ...]"""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ALL = ["1", "2", "3", "4", "5", "6", "7", "A1", "B1"]

todo = sys.argv[1:] or ALL
failed = []
for f in todo:
    script = os.path.join(HERE, f"make_figure{f}.py")
    print(f"\n===== Figure {f} =====", flush=True)
    t0 = time.time()
    rc = subprocess.run([sys.executable, script], cwd=HERE).returncode
    print(f"----- Figure {f}: {'OK' if rc == 0 else 'FAILED'} ({time.time() - t0:.0f}s)", flush=True)
    if rc:
        failed.append(f)
print("\nAll figures OK." if not failed else f"\nFAILED: {', '.join(failed)}")
sys.exit(1 if failed else 0)
