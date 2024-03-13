from dotenv import load_dotenv
import os

os.system("source .env")    # Source the environment file
load_dotenv(dotenv_path=".env")     # Load it into python

# Assign the environment variables to python global variables
SLIDE_N = os.environ.get("SLIDE_N")