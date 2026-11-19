"""Auditable candidate screening for the local English–Hindi fine-tuning probe."""

import math
import re
import unicodedata
from collections import Counter, defaultdict

from src.data.corpus import _grams, comparison_key


def digits(text):
    normalized = ''.join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)
    return Counter(re.findall(r'\d+(?:[.,]\s*\d+)*', normalized.replace(', ', ',').replace('. ', '.')))


def negated(text, language):
    pattern = (r"\b(?:no|not|never|neither|nor|cannot|without)\b|n['’]t\b" if language == 'en'
               else r'(?:^|\s)(?:नहीं|नही|न|ना|मत|बिना|कभी नहीं)(?=\s|$)')
    return bool(re.search(pattern, comparison_key(text) if language == 'hi' else text.casefold()))


class HeldOutIndex:
    """Exact normalized and character-trigram Jaccard checks on either side."""

    def __init__(self, held_out, threshold=0.85):
        self.threshold = threshold
        self.keys = [{comparison_key(row[side]) for row in held_out} for side in ('source', 'target')]
        self.postings = []
        for keys in self.keys:
            postings = defaultdict(list)
            for key in sorted(keys):
                features = frozenset(_grams(key))
                for gram in features:
                    postings[gram].append(features)
            self.postings.append(postings)

    def overlaps(self, text, side):
        key = comparison_key(text)
        if key in self.keys[side]:
            return 'exact'
        features = _grams(key)
        # A match must share at least ceil(threshold * query_size) grams.
        # At least one of this many rare query grams must therefore be present.
        low = math.ceil(self.threshold * len(features))
        high = len(features) / self.threshold
        postings = self.postings[side]
        prefix = sorted(features, key=lambda gram: (len(postings.get(gram, ())), gram))[:len(features) - low + 1]
        candidates = {other for gram in prefix for other in postings.get(gram, ()) if low <= len(other) <= high}
        for other in candidates:
            overlap = len(features & other)
            if overlap / (len(features) + len(other) - overlap) >= self.threshold:
                return 'near'
        return None


def screen(source, target):
    """Conservative surface heuristics; neither acceptance nor rejection proves semantics."""
    src_words = comparison_key(source).split()
    tgt_words = comparison_key(target).split()
    if not 4 <= len(src_words) <= 30 or not 3 <= len(tgt_words) <= 35:
        return 'rejected', ['outside_sentence_length']
    if re.search(r'https?://|<\s*/?\w+\s*>|\b(?:LRB|RRB)\b|%\s*[sd]|\ufffd', source + ' ' + target):
        return 'rejected', ['markup_placeholder_or_corrupt_character']
    letters = [char for char in target if char.isalpha()]
    if not letters or sum('\u0900' <= char <= '\u097f' for char in letters) < 0.9 * len(letters):
        return 'rejected', ['weak_hindi_script_signal']
    reasons = []
    if not source.endswith(('.', '?', '!', '"', '”')) or not target.endswith(('।', '.', '?', '!', '"', '”')):
        reasons.append('missing_sentence_boundary')
    if not 0.55 <= len(tgt_words) / len(src_words) <= 1.8:
        reasons.append('word_ratio')
    if digits(source) != digits(target):
        reasons.append('digit_mismatch_or_spelled_number')
    if negated(source, 'en') != negated(target, 'hi'):
        reasons.append('negation_mismatch_or_paraphrase')
    if re.search(r'\b[A-Z][a-z]+\b|\b[A-Z]{2,}\b', source[1:]):
        reasons.append('entity_needs_bilingual_check')
    if any(a == b == c for a, b, c in zip(tgt_words, tgt_words[1:], tgt_words[2:])):
        reasons.append('repeated_hindi_words')
    return ('uncertain', reasons) if reasons else ('high_confidence_candidate', [])


def domain(source):
    text = source.casefold()
    if re.search(r'\b(?:court|section|government|shall|ministry|scheme|authority)\b', text):
        return 'administrative'
    if re.search(r'\b(?:software|file|server|click|button|system|device)\b', text):
        return 'technical'
    if re.search(r'\b(?:reported|election|police|announced|newspaper)\b', text):
        return 'news_like'
    if re.search(r'\b(?:i|you|we|my|your|please|hello|thanks)\b', text) and len(text.split()) <= 20:
        return 'everyday_proxy'
    return 'general'
