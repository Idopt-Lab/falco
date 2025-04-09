import subprocess
import sys
import os
from flight_simulator import REPO_ROOT_FOLDER

def run_installer():
    path = os.path.join(REPO_ROOT_FOLDER, "installers", "graphviz-install-12.2.1-win64.exe")
    if os.name == 'nt':
        print("Installing Graphviz system binary...")
        subprocess.run([path, '/S'], check=True)  # Silent install
    else:
        print("Graphviz system install skipped (not Windows, install for your system directly off graphviz website)")

def install_conda_env():
    subprocess.check_call(["conda", "env", "update", "--file", "environment.yml", "--prune"])

if __name__ == "__main__":
    run_installer()
    install_conda_env()
