import subprocess
import sys
import os


#import sys
# print(sys.executable)
script = "run_process.py"

path_script = os.path.abspath(script)


with open("out_def.log", "a") as log_file:
   
    subprocess.Popen(
        [sys.executable, path_script],  
        stdout=log_file,
        stderr=log_file,
        stdin=subprocess.DEVNULL,       
        close_fds=True
    )
