from dotenv import load_dotenv
import os
from pathlib import Path
import sys

# Set the root directory to project home path
root_path = Path.cwd()
sys.path.append(str(root_path))

def get_n_value(root_path):
    os.system(f"source {root_path}/.env")    # Source the environment file
    load_dotenv(dotenv_path=f"{root_path}/.env")     # Load it into python

    # Assign the environment variables to python global variables
    SLIDE_N = os.environ.get("SLIDE_N")
    return SLIDE_N

if __name__ == "__main__":
    get_n_value(root_path)