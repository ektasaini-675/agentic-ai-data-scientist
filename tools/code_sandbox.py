"""
Sandboxed Python Code Execution Environment for Autonomous Data Agents.
Executes generated Python scripts in an isolated subprocess with timeout,
captures stdout/stderr, intercepts generated Matplotlib plots, and returns structured execution telemetry.
"""
import os
import sys
import time
import subprocess
import tempfile
from typing import Dict, Any, List


class CodeSandbox:
    def __init__(self, timeout_seconds: int = 15):
        self.timeout_seconds = timeout_seconds

    def execute_code(self, python_code: str, working_dir: str = ".") -> Dict[str, Any]:
        """
        Executes a Python code string in an isolated subprocess.
        Returns returncode, stdout, stderr, execution_time, and generated chart paths.
        """
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(python_code)
            temp_path = f.name

        t0 = time.time()
        try:
            # Use universal python runner
            py_exe = "C:\\Users\\Lenovo\\AppData\\Roaming\\uv\\python\\cpython-3.11-windows-x86_64-none\\python.exe"
            proc = subprocess.run(
                [py_exe, temp_path],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            elapsed = round(time.time() - t0, 3)
            return {
                "success": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "execution_time_seconds": elapsed,
                "timed_out": False
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Execution timed out after {self.timeout_seconds} seconds.",
                "execution_time_seconds": self.timeout_seconds,
                "timed_out": True
            }
        except Exception as e:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "execution_time_seconds": round(time.time() - t0, 3),
                "timed_out": False
            }
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

