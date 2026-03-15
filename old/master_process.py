import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import time
import psutil

path = os.path.abspath(os.getcwd())
base = "C:/Users/lgarc/OneDrive/Escritorio/Universidad/Máster/TFM/Data/"
list_directories = [base + "Triangulos/", base + "Estrellas/"]
# base + "Triangle/",
python_env = r"C:/Users/lgarc/micromamba/envs/hyperspy-env/python.exe"

index_directories = 0


while(index_directories < len(list_directories)):

    currently_running = sum(1 for proc in psutil.process_iter() if proc.name() == 'python.exe')

    dm4_files = [f for f in os.listdir(list_directories[index_directories]) if f.endswith('.dm4')]
    i = 0 
    for file in dm4_files:  
        

        subprocess.Popen([
            python_env,
            "run_process.py",
            str(list_directories[index_directories]),
            file,
            str(i)
        ])
        # subprocess.call("python3 run_process.py '" + str(list_directories[index_directories]) + "' '" + file + "' '" + str(i) + "' &", shell = True)
        currently_running = sum(1 for proc in psutil.process_iter() if proc.name() == 'run_process.py')
        i+= 1

        print(f'File {i} of {len(dm4_files)} in directory {list_directories[index_directories]}')

    time.sleep(10)
    
    index_directories += 1

print("FINISHED!") 