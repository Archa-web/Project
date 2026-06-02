"""
Three-tier bash command safety checker using regex pattern matching.
Classifies any command as BLOCK (never run), WARN (needs confirmation), or SAFE (run freely).
Covers destructive operations like rm -rf /, fork bombs, raw disk writes, and risky sudo usage.
"""

import re
from dataclasses import dataclass

@dataclass
class SafetyResult:
    level:    str    
    message:  str
    command:  str
    can_run:  bool

class SafetyChecker:

    BLOCKED_PATTERNS = [
        (r"rm\s+-[a-z]*r[a-z]*f?\s+/$",            "Deleting root /"),
        (r"rm\s+-[a-z]*rf?\s+/\*",                  "Deleting /* wipes entire system"),
        (r"rm\s+-[a-z]*rf?\s+~$",                   "Deleting home directory"),
        (r"mkfs",                                    "Formatting a filesystem"),
        (r"dd\s+.*of=/dev/[a-z]+\b",                "Writing directly to block device"),
        (r":\(\)\{.*\};:",                           "Fork bomb"),
        (r">\s*/dev/sd[a-z]",                        "Writing to raw disk"),
        (r"chmod\s+-[rR]\s+777\s+/",                "Making entire filesystem world-writable"),
        (r"curl\s+.*\|\s*(?:sudo\s+)?(?:bash|sh)",  "Piping curl to shell"),
        (r"wget\s+.*-O\s*-\s*\|\s*(?:sudo\s+)?(?:bash|sh)", "Piping wget to shell"),
        (r"echo\s+.*>\s*/boot/",                    "Writing to /boot"),
        (r"rm\s+.*vmlinuz",                         "Deleting kernel image"),
        (r"rm\s+.*/etc/passwd",                     "Deleting /etc/passwd"),
        (r"rm\s+.*/etc/shadow",                     "Deleting /etc/shadow"),
        (r"rm\s+.*/etc/sudoers",                    "Deleting /etc/sudoers"),
    ]

    WARN_PATTERNS = [
        (r"rm\s+-[a-z]*rf?\s+\S+",                 "Recursively deletes folder contents"),
        (r"sudo\s+rm\s+",                           "Deleting files as root"),
        (r"shutdown",                               "Shuts down the system"),
        (r"reboot",                                 "Reboots the system"),
        (r"halt|poweroff",                          "Powers off the system"),
        (r"sudo\s+passwd\s+root|passwd\s+root",     "Changing root password"),
        (r"visudo",                                 "Editing sudoers file"),
        (r"chmod\s+777",                            "Setting world-writable permissions"),
        (r">\s*/etc/",                              "Overwriting a file in /etc"),
        (r"sudo\s+apt\s+(?:remove|purge|autoremove)","Removing system packages"),
        (r"pkill\s+-9|killall\s+-9",               "Force-killing processes"),
        (r"crontab\s+-r",                           "Removing all cron jobs"),
        (r"iptables\s+-F|ufw\s+--force\s+reset",   "Flushing firewall rules"),
        (r"dd\s+if=",                               "Using dd — double check target"),
    ]

    def __init__(self):
        self._block_re = [(re.compile(p, re.IGNORECASE), r) for p, r in self.BLOCKED_PATTERNS]
        self._warn_re  = [(re.compile(p, re.IGNORECASE), r) for p, r in self.WARN_PATTERNS]

    def check(self, command: str) -> SafetyResult:
        cmd = command.strip()
        for pattern, reason in self._block_re:
            if pattern.search(cmd):
                return SafetyResult(
                    level   = "block",
                    message = f"🚫 BLOCKED — {reason}",
                    command = cmd,
                    can_run = False,
                )
        for pattern, reason in self._warn_re:
            if pattern.search(cmd):
                return SafetyResult(
                    level   = "warn",
                    message = f"⚠️  WARNING — {reason}",
                    command = cmd,
                    can_run = False,
                )
        return SafetyResult(level="safe", message="", command=cmd, can_run=True)
