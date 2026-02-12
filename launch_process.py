import subprocess
import sys
import os


script = "run_process.py"

path_script = os.path.abspath(script)


with open("out.log", "a") as log_file:
    subprocess.Popen(
        [sys.executable, path_script],  
        stdout=log_file,
        stderr=log_file,
        stdin=subprocess.DEVNULL,       
        close_fds=True
    )

print(f"Programa '{script}' lanzado en background. Revisa 'out.log' para la salida.")
