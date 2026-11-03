#!/usr/bin/env python3
"""Envoie une suite de commandes UCI avec des pauses et horodate les reponses."""
import subprocess
import sys
import threading
import time

engine = sys.argv[1]
script = sys.argv[2:]  # elements : "cmd" ou "sleep:secondes"

p = subprocess.Popen([engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
t0 = time.time()


def reader():
    for line in p.stdout:
        if line.startswith(("bestmove", "info string", "readyok")) or "depth 1 " in line:
            print(f"{time.time() - t0:7.3f}s  < {line.rstrip()}")


th = threading.Thread(target=reader, daemon=True)
th.start()
for item in script:
    if item.startswith("sleep:"):
        time.sleep(float(item[6:]))
    else:
        print(f"{time.time() - t0:7.3f}s  > {item}")
        p.stdin.write(item + "\n")
        p.stdin.flush()
time.sleep(0.5)
try:
    p.stdin.write("quit\n")
    p.stdin.flush()
except BrokenPipeError:
    pass
p.wait(timeout=10)
th.join(timeout=1)
print("code de sortie", p.returncode)
