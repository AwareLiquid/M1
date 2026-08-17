import subprocess, sys, os
print("Step 1: Installing torch")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch==2.4.1', '--index-url', 'https://download.pytorch.org/whl/cu121'], check=True)
print("Step 2: Cloning M1")
subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/AwareLiquid/M1.git', '/kaggle/working/M1'], check=True)
os.chdir('/kaggle/working/M1')
print("Step 3: Installing dependencies")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt', 'peft', 'datasets'], check=True)
print("Done! Check if all imports work next.")
