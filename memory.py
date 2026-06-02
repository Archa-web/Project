"""
SBERT-based vector memory for RAG-style bash command retrieval.
Encodes a JSONL dataset of (instruction, command) pairs using all-MiniLM-L6-v2,
caches the index to disk, and retrieves the top-k most similar examples at query time.
Supports online learning by appending new confirmed pairs to the dataset.
"""

import json
import logging
import os
import torch
from sentence_transformers import SentenceTransformer, util

class TerminalMemory:
    def __init__(self, jsonl_path: str, index_path: str = "vector_index.pt"):
        self.jsonl_path  = jsonl_path
        self.index_path  = index_path
        self.encoder     = SentenceTransformer("all-MiniLM-L6-v2")
        self.prompts:  list[str] = []
        self.commands: list[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_jsonl()
        self._load_or_build_index()

    def _load_jsonl(self):
        with open(self.jsonl_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    inp  = data.get("input", "").strip()
                    out  = data.get("output", "").strip()
                    if inp and out:
                        self.prompts.append(inp)
                        self.commands.append(out)
                except:
                    continue

    def _load_or_build_index(self):
        if os.path.exists(self.index_path):
            self.encoded_prompts = torch.load(
                self.index_path,
                map_location=torch.device("cpu"),
                weights_only=True,
            ).to(self.device)
        else:
            print(f"[System] Encoding {len(self.prompts)} commands, please wait...")
            self.encoded_prompts = self.encoder.encode(
                self.prompts, convert_to_tensor=True
            )
            torch.save(self.encoded_prompts.cpu(), self.index_path)
            self.encoded_prompts = self.encoded_prompts.to(self.device)

    def get_context(self, user_input: str, k: int = 3) -> list[dict]:
        
        user_vector = self.encoder.encode(user_input, convert_to_tensor=True).to(self.device)
        scores      = util.cos_sim(user_vector, self.encoded_prompts)[0]
        top_results = torch.topk(scores, k=min(k, len(self.prompts)))

        matches = []
        for score, idx in zip(top_results.values, top_results.indices):
            matches.append({
                "prompt": self.prompts[idx],
                "cmd":    self.commands[idx],
                "score":  round(float(score), 4),
            })

        if matches:
            logging.debug(f"[SBERT] Top match: {matches[0]['score']:.2f} — {matches[0]['cmd']!r}")

        return matches

    def learn(self, user_input: str, cmd: str) -> bool:
        
        user_vector = self.encoder.encode(user_input, convert_to_tensor=True).to(self.device)
        scores      = util.cos_sim(user_vector, self.encoded_prompts)[0]
        top_score   = float(scores.max())

        if top_score > 0.97:
            logging.debug(f"[Memory] Skipped (duplicate, score {top_score:.2f}): {user_input!r}")
            return False

        self.prompts.append(user_input)
        self.commands.append(cmd)

        new_vector           = self.encoder.encode(user_input, convert_to_tensor=True).to(self.device)
        self.encoded_prompts = torch.cat([self.encoded_prompts, new_vector.unsqueeze(0)], dim=0)

        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps({"input": user_input, "output": cmd}) + "\n")

        torch.save(self.encoded_prompts.cpu(), self.index_path)
        logging.debug(f"[Memory] Learned: {user_input!r} → {cmd!r} (index size: {len(self.prompts)})")
        return True