from dotenv import load_dotenv
import os

cwd = os.getcwd()
env_path = f"{cwd}/.env"

if os.path.exists(env_path):
    # Source the environment file and load it into python
    os.system(f"source {env_path}")
    load_dotenv(dotenv_path=f"{env_path}")

else:
    env_path = f"{os.path.dirname(cwd)}/.env"
    # Source the environment file and load it into python
    os.system(f"source {env_path}")
    load_dotenv(dotenv_path=f"{env_path}")

SLIDE_N = os.environ.get("SLIDE_N")
ESP_OUTPUT_FOLDER = os.environ.get("ESP_OUTPUT_FOLDER")
DAILY_OUTPUT_FOLDER = os.environ.get("DAILY_OUTPUT_FOLDER")
RESAMPLING_FREQ = os.environ.get("RESAMPLING_FREQ")