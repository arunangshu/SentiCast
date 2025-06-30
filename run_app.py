import subprocess
import os
import sys

def main():
    """Run the Streamlit app with the correct Python path"""
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run the Streamlit app with the current directory in PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = current_dir
    
    # Command to run the app
    cmd = [sys.executable, "-m", "streamlit", "run", os.path.join(current_dir, "app", "main.py")]
    
    # Execute the command
    subprocess.call(cmd, env=env)

if __name__ == "__main__":
    main() 