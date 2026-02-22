import subprocess
import sys
import os
import datetime


script = "run_process.py"

path_script = os.path.abspath(script)



with open("out.log", "a") as log_file:
    print(f'{script} lanzado en {datetime.datetime.now()}. Revisa out.log para la salida.')
    subprocess.Popen(
        [sys.executable, path_script],  
        stdout=log_file,
        stderr=log_file,
        stdin=subprocess.DEVNULL,       
        close_fds=True
    )

