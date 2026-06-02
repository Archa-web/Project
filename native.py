"""
Translates Manglish (romanized Malayalam) input into English as a pipeline preprocessing step.
Uses ai4bharat XlitEngine to convert Manglish to Malayalam script, then facebook/nllb-200
to translate to English. Detects Malayalam input via Unicode check, lingua, or keyword fallback.
"""

import argparse
import torch
torch.serialization.add_safe_globals([argparse.Namespace])
from ai4bharat.transliteration import XlitEngine
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

class MalayalamTranslator:
    
    _MANGLISH_HINTS = {
        # Verbs
        "undakku", "undak", "undoo",
        "cheyyuka", "cheyyu", "cheyyoo", "cheyy", "cheyyaan", "cheyyaam",
        "kanikku", "kaanikkan", "nokku", "nokkoo", "nokkaan", "nokkaam",
        "tirakkuka", "thurakku", "adukkuka", "maykuka",
        "edukku", "edoo", "edu", "kodukkoo", "kodukku",
        "parayu", "tharu", "thaa", "varu", "varoo", "varaam",
        "kanaan", "kaanaan", "poyi", "pokaam",
        "cheyyan", "nokkan",
        # Pronouns / particles
        "njan", "ningal", "avan", "aval", "avante",
        "ente", "ninte", "oru", "athu", "ithu", "ethu",
        "aanu", "illa", "alle", "athe", "okke",
        # Question words
        "enthanu", "enthaa", "enth", "entha", "enthokke",
        "evidanu", "evide", "evidey", "evidaanu",
        "engane", "angane", "ingane", "eppol",
        # Common words
        "upayogam", "sahaayam", "venam", "venda", "veno",
        "peril", "ennoru", "thora", "ittu",
        "adipoli", "pinne", "sheriyaa", "shari",
        "sheriyalla", "ariyilla", "manasilaayi",
        "veedu", "padam", "vannu",
        "tharam", "tharoo", "pero", "peru",
    }

    _ENGLISH_STOPWORDS = {
        "the", "a", "an", "in", "on", "at", "to", "for", "of",
        "and", "or", "not", "with", "from", "into", "out",
        "up", "down", "here", "there", "this", "that",
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "will", "would", "should", "could", "can", "may", "might",
        "my", "your", "his", "her", "its", "our", "their",
        "what", "where", "when", "how", "which", "who", "why",
        "please", "hey", "hi", "hello",
        "new", "old", "all", "some", "any", "no",
        "get", "set", "run", "go", "put", "give", "take",
        "show", "find", "make", "use", "add", "see",
        "help", "want", "need", "try",
    }

    def __init__(self):
        print("[Malayalam] Loading transliteration engine...")
        self.xlit = XlitEngine("ml", beam_width=5)

        print("[Malayalam] Loading NLLB translation model...")
        model_id        = "facebook/nllb-200-distilled-600M"
        self.tokenizer  = NllbTokenizer.from_pretrained(model_id)
        self.model      = AutoModelForSeq2SeqLM.from_pretrained(model_id).to("cpu")
        self.english_id = self.tokenizer.convert_tokens_to_ids("eng_Latn")

        self._detector     = None
        self._mal_language = None
        try:
            from lingua import Language, LanguageDetectorBuilder
            self._mal_language = Language.MALAYALAM
            self._detector = LanguageDetectorBuilder.from_languages(
                Language.ENGLISH, Language.MALAYALAM
            ).build()
            print("[Malayalam] lingua detector ready.")
        except ImportError:
            print("[Malayalam] lingua not installed — using keyword fallback.")
        except Exception as e:
            print(f"[Malayalam] lingua init failed ({e}) — using keyword fallback.")

        print("[Malayalam] Ready.")

    def is_manglish(self, text: str) -> bool:
       
        if any('\u0d00' <= c <= '\u0d7f' for c in text):
            return True

        if self._detector is not None:
            try:
                result = self._detector.detect_language_of(text)
                if result == self._mal_language:
                    return True
                confidences = self._detector.compute_language_confidence_values(text)
                for conf in confidences:
                    if conf.language == self._mal_language and conf.value > 0.4:
                        return True
            except Exception:
                pass

        words         = set(text.lower().split())
        manglish_hits = words & self._MANGLISH_HINTS
        english_hits  = words & self._ENGLISH_STOPWORDS

        if not manglish_hits:
            return False

        pure_english = len(english_hits) >= len(words) * 0.8

        return len(manglish_hits) >= 2 or (len(manglish_hits) >= 1 and not pure_english)

    def translate(self, text: str) -> tuple[str, str]:
        mal_script = self.xlit.translit_sentence(text)['ml']
        self.tokenizer.src_lang = "mal_Mlym"
        inputs = self.tokenizer(mal_script, return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        with torch.no_grad():
            tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.english_id,
                max_length=100,
            )
        english = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        return mal_script, english
