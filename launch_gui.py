#!/usr/bin/env python
"""
Launcher script for Paper Airplane AI Optimizer GUI
Run from project root: python launch_gui.py
"""
import subprocess
import sys
from pathlib import Path
import webbrowser
import time

def main():
    project_root = Path(__file__).parent
    
    print("\n" + "="*50)
    print("  Paper Airplane AI Optimizer - GUI")
    print("="*50 + "\n")
    
    print("Starting Streamlit app...")
    print("Opening browser at http://localhost:8502\n")
    
    # Give browser time to open after app starts
    def open_browser():
        time.sleep(3)
        webbrowser.open("http://localhost:8502")
    
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Run streamlit
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "src/gui/app.py"],
        cwd=project_root
    )

if __name__ == "__main__":
    main()
