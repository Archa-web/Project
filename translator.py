"""
Translates plain-English instructions into bash commands using a fine-tuned T5 model.
Supports RAG-augmented prompting, filename/path entity injection, and automatic flag inference.
The CorrectedTerminalTranslator subclass adds Ollama-based correction as a final pass.
"""

import re
import logging
import shlex
import subprocess
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

_KNOWN_CMDS = {
    "ls", "cat", "cp", "mv", "rm", "mkdir", "rmdir", "touch", "echo", "pwd",
    "cd", "find", "grep", "awk", "sed", "sort", "uniq", "wc", "head", "tail",
    "less", "more", "man", "which", "whereis", "type", "file", "stat", "du",
    "df", "ps", "top", "kill", "killall", "jobs", "bg", "fg", "nohup",
    "chmod", "chown", "chgrp", "ln", "readlink", "tar", "zip", "unzip",
    "gzip", "gunzip", "curl", "wget", "ssh", "scp", "rsync", "ping",
    "ifconfig", "ip", "netstat", "ss", "hostname", "uname", "whoami", "id",
    "useradd", "usermod", "passwd", "su", "sudo", "env", "export", "source",
    "alias", "history", "clear", "date", "cal", "uptime", "free", "lscpu",
    "lsblk", "mount", "umount", "dd", "mkfs", "fdisk", "lsof", "strace",
    "diff", "patch", "make", "gcc", "python", "python3", "pip", "pip3",
    "git", "docker", "systemctl", "service", "journalctl", "cron", "crontab",
    "code", "code-insiders", "subl", "gedit", "xdg-open", "htop", "fzf", "tree",
    "xargs", "tee", "tr", "cut", "paste", "join", "split", "nl", "od",
    "xxd", "base64", "md5sum", "sha256sum", "openssl", "nano", "vim", "vi",
}

RAG_HIGH_CONFIDENCE = 0.85
RAG_MED_CONFIDENCE  = 0.60
T5_BEAM_CONFIDENCE  = -0.5

_FLAG_RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bhidden\b",          re.I), "ls",   "-la"),
    (re.compile(r"\brecursive(ly)?\b",  re.I), "ls",   "-R"),
    (re.compile(r"\bdetail(s|ed)?\b",   re.I), "ls",   "-lh"),
    (re.compile(r"\bsize\b",            re.I), "ls",   "-lS"),
    (re.compile(r"\bnewest|latest\b",   re.I), "ls",   "-lt"),
    (re.compile(r"\bhuman.?readable\b", re.I), "ls",   "-lh"),
    (re.compile(r"\bignore.?case\b",    re.I), "grep", "-i"),
    (re.compile(r"\brecursive(ly)?\b",  re.I), "grep", "-r"),
    (re.compile(r"\bline.?number\b",    re.I), "grep", "-n"),
    (re.compile(r"\bcount\b",           re.I), "grep", "-c"),
    (re.compile(r"\binvert\b",          re.I), "grep", "-v"),
    (re.compile(r"\bforce\b",           re.I), "rm",   "-f"),
    (re.compile(r"\bverbose\b",         re.I), "cp",   "-v"),
    (re.compile(r"\brecursive(ly)?\b",  re.I), "cp",   "-r"),
    (re.compile(r"\bhuman.?readable\b", re.I), "du",   "-sh"),
    (re.compile(r"\bhuman.?readable\b", re.I), "df",   "-h"),
    (re.compile(r"\bextract\b",         re.I), "tar",  "-xzf"),
    (re.compile(r"\bcompress\b",        re.I), "tar",  "-czf"),
]

def _extract_entities(user_input: str) -> list[str]:
    _IGNORE = {
        "python", "python3", "bash", "shell", "linux", "file", "files",
        "folder", "directory", "current", "only", "all", "the", "in",
        "show", "list", "find", "get", "display", "print",
    }
    quoted = re.findall(r'["\'\']+([^"\'\']+)["\'\']+', user_input)
    if quoted:
        return quoted
    tokens = user_input.split()
    path_like = [
        t for t in tokens
        if ("." in t or "/" in t or t.startswith("~"))
        and t.lower() not in _IGNORE
    ]
    return path_like if path_like else []

def _inject_flags(cmd: str, user_input: str) -> str:
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return cmd
    if not tokens:
        return cmd
    base     = tokens[0]
    existing = {t for t in tokens if t.startswith("-")}
    extra    = []
    for pattern, target, flag in _FLAG_RULES:
        if target == base and pattern.search(user_input) and flag not in existing:
            extra.append(flag)
            existing.add(flag)
    return " ".join([base] + extra + tokens[1:]) if extra else cmd

def _is_valid_cmd(cmd: str) -> bool:
    if not cmd or not cmd.strip():
        return False
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return False
    if not tokens or tokens.count(tokens[0]) > 1:
        return False
    base = tokens[0]
    if base in _KNOWN_CMDS:
        return True
    return subprocess.run(["which", base], capture_output=True, text=True).returncode == 0

def _clean(raw: str) -> str:
    import re as _re
    raw = _re.split(r'[\s]+(?:Exemple|Example|input|output)[\s]*[:>]', raw, flags=_re.IGNORECASE)[0]
    raw = _re.sub(r'[\s]*->[\s]*.*', '', raw)
    for artifact in ("Target:", "translate English to Bash:", "|"):
        raw = raw.replace(artifact, "")
    
    try:
        tokens = shlex.split(raw.strip())
    except ValueError:
        return raw.strip()
    if not tokens:
        return raw.strip()
    seen = []
    for tok in tokens:
        if tok == tokens[0] and seen:
            break
        seen.append(tok)
    return " ".join(seen)

class TerminalTranslator:
    def __init__(self, model_path="./fine_tuned_nlt_v3"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, use_fast=False)
        self.model     = T5ForConditionalGeneration.from_pretrained(model_path)
        self.device    = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def _build_prompt(self, user_input: str, rag_matches: list | None) -> str:
        if rag_matches:
            examples = "\n".join(f"Example: {m['prompt']} -> {m['cmd']}" for m in rag_matches)
            return f"{examples}\ntranslate English to Bash: {user_input}"
        return f"translate English to Bash: {user_input}"

    def generate_candidates(
        self,
        user_input:  str,
        rag_matches: list | None = None,
        n:           int = 2,
        **kwargs,
    ) -> tuple[list[str], list[float]]:
        entities  = _extract_entities(user_input)
        num_beams = max(n + 2, 5)

        prompt = self._build_prompt(user_input, rag_matches)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=256, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=64,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        raw_scores = outputs.sequences_scores.tolist() if hasattr(outputs, "sequences_scores") else [0.0] * num_beams
        results, beam_scores = [], []

        for seq, score in zip(outputs.sequences, raw_scores):
            cmd = _clean(self.tokenizer.decode(seq, skip_special_tokens=True))
        
            cmd = _inject_flags(cmd, user_input)
            if entities and not any(e in cmd for e in entities):
                cmd = cmd + " " + shlex.quote(entities[0])
            if cmd and cmd not in results:
                results.append(cmd)
                beam_scores.append(score)
            if len(results) >= n:
                break

        valid = [(c, s) for c, s in zip(results, beam_scores) if _is_valid_cmd(c)]
        if valid and rag_matches:
            rag_bases = {m["cmd"].split()[0] for m in rag_matches}
            preferred = [(c, s) for c, s in valid if c.split()[0] in rag_bases]
            rest      = [(c, s) for c, s in valid if c.split()[0] not in rag_bases]
            valid     = preferred + rest

        
        if not valid and rag_matches:
            fallback = rag_matches[0]["cmd"]
            logging.debug(f"[Translator] RAG fallback: {fallback!r}")
            valid = [(fallback, -999.0)]

        valid = valid or [(c, s) for c, s in zip(results, beam_scores)]
        return [c for c, _ in valid], [s for _, s in valid]

class CorrectedTerminalTranslator(TerminalTranslator):
    def __init__(self, model_path: str = "./fine_tuned_nlt_v3", api_key: str | None = None):
        super().__init__(model_path)
        from corrector import CommandCorrector
        self.corrector       = CommandCorrector(api_key=api_key)
        self.descriptions:   dict[str, str] = {}
        self.pipeline_source: str = ""

    def generate_candidates(
        self,
        user_input:  str,
        rag_matches: list | None = None,
        n:           int = 2,
    ) -> list[str]:
        self.descriptions    = {}
        self.pipeline_source = ""
        rag           = rag_matches or []
        top_rag_score = rag[0].get("score", 0.0) if rag else 0.0

        if top_rag_score >= RAG_HIGH_CONFIDENCE:
            best_rag = rag[0]["cmd"]
            logging.debug(f"[Pipeline] RAG high ({top_rag_score:.2f}) — skipping T5")
            self.pipeline_source = f"RAG ({top_rag_score:.2f})"
            pairs, _ = self.corrector.correct(user_input, best_rag, rag)
            for cmd, desc in pairs:
                self.descriptions[cmd] = desc
            return [cmd for cmd, _ in pairs]

        context = rag if top_rag_score >= RAG_MED_CONFIDENCE else None
        self.pipeline_source = f"T5+RAG ({top_rag_score:.2f})" if context else "T5"
        logging.debug(f"[Pipeline] {self.pipeline_source}")

        candidates, beam_scores = super().generate_candidates(user_input, context, n)
        if not candidates:
            return []

        best       = candidates[0]
        best_score = beam_scores[0] if beam_scores else -999.0

        if not self.corrector.enabled:
            return candidates

        logging.debug(f"[Pipeline] T5 beam score {best_score:.2f} — {'correcting' if best_score < T5_BEAM_CONFIDENCE else 'describing'}")
        pairs, was_corrected = self.corrector.correct(user_input, best, rag)
        for cmd, desc in pairs:
            self.descriptions[cmd] = desc
        if was_corrected:
            logging.debug(f"[Corrector] T5: {best!r} → Ollama: {pairs[0][0]!r}")
        return [cmd for cmd, _ in pairs]