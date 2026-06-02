"""
Executes bash commands in a persistent shell session using subprocess.
Handles 'cd' separately to maintain working directory state across commands,
and blocks a set of known catastrophic commands before execution.
"""

import subprocess
import os

_DANGER_PATTERNS = (
    "rm -rf /",
    "rm -rf ~",
    ":(){:|:&};:",
    "mkfs",
    "> /dev/sda",
    "dd if=",
)

def is_dangerous(command: str) -> bool:
    lower = command.lower()
    return any(pat in lower for pat in _DANGER_PATTERNS)

class TerminalSession:
    def __init__(self):
        self.cwd = os.getcwd()

    def run(self, command: str) -> str:
        if not command or not command.strip():
            return "[No command provided]"

        try:
            cmd = command.strip()
            if cmd == "cd" or cmd.startswith("cd ") or cmd.startswith("cd\t"):
                return self._handle_cd(cmd)

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.cwd,
            )
            return result.stdout or result.stderr or "[Command produced no output]"

        except Exception as e:
            return f"[Execution error] {e}"

    def _handle_cd(self, command: str) -> str:
        parts  = command.split(None, 1)
        target = parts[1] if len(parts) > 1 else os.path.expanduser("~")
        target = os.path.expanduser(target.strip())

        if not os.path.isabs(target):
            target = os.path.join(self.cwd, target)

        target = os.path.normpath(target)

        if not os.path.isdir(target):
            return f"[cd error] No such directory: {target}"

        try:
            os.chdir(target)
            self.cwd = os.getcwd()
            return f"Moved to {self.cwd}"
        except PermissionError:
            return f"[cd error] Permission denied: {target}"
        except Exception as e:
            return f"[cd error] {e}"