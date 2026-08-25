"""Rule-based content screener — fast first-pass detection.

Three tiers:
  1. Unicode / encoding tricks  (zero-width, tag-block, homoglyphs)
  2. Prompt-injection phrases    (regex over known attack signatures)
  3. Structural anomalies        (special-char ratios, instruction-verb density)

Returns a score in [0, 1] and a list of matched rule names.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class ScreenResult:
    score: float                         # [0, 1]
    matched_rules: list[str] = field(default_factory=list)
    details: dict[str, float] = field(default_factory=dict)


class RuleScreener:
    """Fast heuristic screener.

    Each rule contributes a weight toward the total score.  The total is
    clipped to [0, 1] so the combiner can treat it as a probability.
    """

    # ------------------------------------------------------------------
    # Rule weights (tuned empirically — see eval/report.ipynb)
    # ------------------------------------------------------------------
    RULE_WEIGHTS: dict[str, float] = {
        # Tier 1: encoding tricks
        "zero_width_chars":       0.60,
        "unicode_tag_block":      0.90,
        "homoglyph_chars":        0.45,
        "html_comment_in_text":   0.30,
        "base64_blob":            0.35,
        # Tier 2: injection phrases
        "ignore_instruction":     0.85,
        "system_role_marker":     0.80,
        "assistant_role_marker":  0.75,
        "disregard_previous":     0.85,
        "reveal_prompt":          0.80,
        "act_as_instruction":     0.70,
        "jailbreak_phrase":       0.85,
        "command_injection":      0.75,
        # Tier 3: structural
        "high_special_char_ratio": 0.30,
        "high_instruction_verb_density": 0.50,
        "suspicious_formatting":  0.25,
    }

    # Tier 2: compiled injection-phrase patterns
    _INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
        (
            "ignore_instruction",
            re.compile(
                r"(ignore|disregard|forget|bypass|override|suppress)\s+(all\s+)?"
                r"(previous|prior|above|earlier|old|your|the)?\s*(instruction|prompt|directive|rule|constraint|system)",
                re.IGNORECASE,
            ),
        ),
        (
            "system_role_marker",
            re.compile(r"(^|\n)\s*<<?\s*system\s*>>?", re.IGNORECASE),
        ),
        (
            "assistant_role_marker",
            re.compile(r"(^|\n)\s*<<?\s*(assistant|user|human)\s*>>?", re.IGNORECASE),
        ),
        (
            "disregard_previous",
            re.compile(
                r"(do\s+not|don'?t|never)\s+(follow|obey|respect|adhere\s+to)\s+(the|your|these)",
                re.IGNORECASE,
            ),
        ),
        (
            "reveal_prompt",
            re.compile(
                r"(print|output|reveal|show|display|repeat|tell\s+me)\s+(your\s+)?(system\s+)?"
                r"(prompt|instruction|directive|rules?)",
                re.IGNORECASE,
            ),
        ),
        (
            "act_as_instruction",
            re.compile(
                r"(act|behave|pretend|roleplay|you\s+are\s+now)\s+(as|like)\s+(a\s+)?(dan|jailbreak|unrestricted|evil|bad|malicious)",
                re.IGNORECASE,
            ),
        ),
        (
            "jailbreak_phrase",
            re.compile(
                r"\b(DAN|jailbreak|do anything now|no\s+restrictions|no\s+limits|unfiltered)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "command_injection",
            re.compile(
                r"(;\s*(rm|del|drop|shutdown|exec|eval|os\.|subprocess)\b|"
                r"\|\s*(bash|sh|cmd|powershell)\b)",
                re.IGNORECASE,
            ),
        ),
    ]

    # Instruction verbs that raise density score
    _INSTRUCTION_VERBS = re.compile(
        r"\b(ignore|forget|disregard|override|bypass|follow|obey|execute|run|print|output|reveal|show|"
        r"repeat|tell|give|provide|act|pretend|roleplay|say|write|generate|produce|return)\b",
        re.IGNORECASE,
    )

    def screen(self, text: str) -> ScreenResult:
        """Run all rule tiers on text and return a ScreenResult."""
        matched: list[str] = []
        details: dict[str, float] = {}
        total_weight = 0.0

        # ------------------------------------------------------------------
        # Tier 1: encoding anomalies
        # ------------------------------------------------------------------
        encoding_flags = self._encoding_checks(text)
        for flag in encoding_flags:
            rule_name = flag.split(":")[0]
            w = self.RULE_WEIGHTS.get(rule_name, 0.0)
            matched.append(flag)
            details[rule_name] = w
            total_weight += w

        # ------------------------------------------------------------------
        # Tier 2: injection phrases
        # ------------------------------------------------------------------
        for rule_name, pattern in self._INJECTION_PATTERNS:
            if pattern.search(text):
                w = self.RULE_WEIGHTS[rule_name]
                matched.append(rule_name)
                details[rule_name] = w
                total_weight += w

        # ------------------------------------------------------------------
        # Tier 3: structural features
        # ------------------------------------------------------------------
        struct_scores = self._structural_checks(text)
        for rule_name, raw_val in struct_scores.items():
            if raw_val > 0:
                w = self.RULE_WEIGHTS.get(rule_name, 0.0) * raw_val
                matched.append(rule_name)
                details[rule_name] = w
                total_weight += w

        # Clip to [0, 1]
        score = min(total_weight, 1.0)
        return ScreenResult(score=score, matched_rules=matched, details=details)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encoding_checks(self, text: str) -> list[str]:
        """Return list of encoding anomaly flag strings (same format as metadata.py)."""
        issues: list[str] = []

        zw = re.findall(
            r"[\u200B\u200C\u200D\u200E\u200F\u2060\u2061\u2062\u2063\uFEFF]", text
        )
        if zw:
            issues.append(f"zero_width_chars:{len(zw)}")

        tag = re.findall(r"[\U000E0000-\U000E007F]", text)
        if tag:
            issues.append(f"unicode_tag_block:{len(tag)}")

        hg = re.findall(r"[\u0400-\u04FF\u0370-\u03FF]", text)
        if len(hg) > 5:
            issues.append(f"homoglyph_chars:{len(hg)}")

        if re.search(r"<!--", text):
            issues.append("html_comment_in_text")

        b64 = re.findall(r"[A-Za-z0-9+/]{100,}={0,2}", text)
        if b64:
            issues.append(f"base64_blob:{len(b64)}")

        return issues

    def _structural_checks(self, text: str) -> dict[str, float]:
        """Return structural anomaly scores (0 = clean, >0 = suspicious)."""
        results: dict[str, float] = {}
        if not text:
            return results

        words = text.split()
        n_words = max(len(words), 1)
        n_chars = max(len(text), 1)

        # Special-char ratio: >, <, |, {, }, ; in regular prose is suspicious
        special = sum(1 for c in text if c in "<>|{}[];=`~^")
        special_ratio = special / n_chars
        results["high_special_char_ratio"] = float(special_ratio > 0.05)

        # Instruction-verb density
        verb_hits = len(self._INSTRUCTION_VERBS.findall(text))
        verb_density = verb_hits / n_words
        results["high_instruction_verb_density"] = float(min(verb_density * 5, 1.0))

        # Suspicious formatting: lots of ALL_CAPS words
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
        results["suspicious_formatting"] = float(caps_words / n_words > 0.15)

        return results
