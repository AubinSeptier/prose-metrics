"""Stylistic and grammatical distribution metrics calculation module for text."""

from collections import Counter
from typing import Final

from spacy.tokens import Doc, Token

from prose_metrics.models.report import StyleMetrics

# Suffixes identifying manner adverbs, supporting English and French
_MANNER_ADVERB_SUFFIXES: Final[dict[str, str]] = {
    "en": "ly",
    "fr": "ment",
}

_IRREGULAR_MANNER_ADVERBS: Final[dict[str, frozenset[str]]] = {
    "en": frozenset(
        {
            "well",
            "fast",
            "hard",
            "late",
            "early",
            "straight",
            "wrong",
            "wrongly",
            "far",
            "high",
            "long",
            "low",
            "near",
        }
    ),
    "fr": frozenset(
        {
            "bien",
            "mal",
            "mieux",
            "pis",
            "vite",
            "peu",
            "clair",
            "fort",
        }
    ),
}

_NON_MANNER_ADVERBS: Final[dict[str, frozenset[str]]] = {
    "en": frozenset(
        {
            "hourly",
            "daily",
            "weekly",
            "monthly",
            "yearly",
            "annually",
            "nightly",
            "fornightly",
            "recently",
            "formerly",
            "previously",
            "lately",
            "eventually",
            "finally",
            "initially",
            "subsequently",
            "quarterly",
            "rarely",
            "usually",
            "occasionally",
            "completely",
            "totally",
            "utterly",
            "absolutely",
            "extremely",
            "highly",
            "intensely",
            "perfectly",
            "thoroughly",
            "fairly",
            "partially",
            "relatively",
            "considerably",
            "significantly",
            "nearly",
            "practically",
            "virtually",
            "roughly",
            "approximately",
            "basically",
            "generally",
            "typically",
            "reportedly",
            "allegedly",
            "arguably",
            "briefly",
            "incidentally",
            "theoretically",
            "technically",
            "naturally",
            "thankfully",
            "regrettably",
            "surprisingly",
            "evidently",
            "hardly",
            "really",
            "currently",
            "mostly",
        }
    ),
    "fr": frozenset(
        {
            "actuellement",
            "habituellement",
            "fréquemment",
            "rarement",
            "récemment",
            "complètement",
            "énormément",
            "fortement",
            "largement",
            "légèrement",
            "abondamment",
            "heureusement",
            "malheureusement",
            "assurément",
            "certainement",
            "probablement",
            "également",
            "autrement",
            "précisément",
        }
    ),
}

# Avoids ty complaints about a bare frozenset()
_EMPTY_WORD_SET: Final[frozenset[str]] = frozenset()


def _modifies_verb(token: Token) -> bool:
    """Determine whether a token syntactically modifies a verb.

    Coordinated adverbs such as "quickly and quietly" are walked back through their conjunction head so every member
    is attributed to the verb that the first member modifies.

    Args:
        token (Token): The spaCy token to inspect.

    Returns:
        bool: True if the token modifies a verb, directly or through coordination, False otherwise.
    """
    current: Token = token
    visited: set[int] = set()

    while current.dep_ == "conj" and current.head.pos_ == "ADV" and current.i not in visited:
        visited.add(current.i)
        current = current.head
    return current.dep_ == "advmod" and current.head.pos_ == "VERB"


def _is_manner_adverb(token: Token, language: str) -> bool:
    """Determine whether a token is a manner adverb.

    Combines three filters: a syntactic check that the token modifies a verb, a lemma lookup for
    irregular manner adverbs, and a suffix check against a stoplist of time and intensity adverbs.
    If unknown language, it yields zero manner adverbs rather than guessing.

    Args:
        token (Token): The spaCy token to inspect.
        language (str): The language code of the pipeline (ISO 639-1, e.g., "en" for English).

    Returns:
        bool: True if the token is classified as a manner adverb, False otherwise.
    """
    if not _modifies_verb(token=token):
        return False

    lemma: str = token.lemma_.lower()
    if lemma in _IRREGULAR_MANNER_ADVERBS.get(language, _EMPTY_WORD_SET):
        return True

    suffix: str | None = _MANNER_ADVERB_SUFFIXES.get(language)
    if suffix is None or token.pos_ != "ADV":
        return False

    text: str = token.text.lower()
    return text.endswith(suffix) and lemma not in _NON_MANNER_ADVERBS.get(language, _EMPTY_WORD_SET)


def compute_style_metrics(doc: Doc) -> StyleMetrics:
    """Compute stylistic and grammatical distribution metrics from a parsed spaCy Doc.

    Args:
        doc (Doc): The parsed spaCy Doc instance.

    Returns:
        StyleMetrics: An instance containing POS ratios, manner adverbs count, and tag distribution.
    """
    language = doc.lang_
    pos_counter: Counter[str] = Counter()
    total_words = 0
    adverbs_manner_count = 0

    for token in doc:
        if token.is_punct or token.is_space:
            continue

        total_words += 1
        pos = token.pos_
        pos_counter[pos] += 1

        # Manner adverbs counting
        if _is_manner_adverb(token=token, language=language):
            adverbs_manner_count += 1

    if total_words == 0:
        return StyleMetrics(
            noun_ratio=0.0,
            verb_ratio=0.0,
            adjective_ratio=0.0,
            adverb_ratio=0.0,
            pronoun_ratio=0.0,
            adverbs_manner_count=0,
            pos_distribution={},
        )

    # Calculate thematic grammar counts
    noun_count = pos_counter.get("NOUN", 0) + pos_counter.get("PROPN", 0)
    verb_count = pos_counter.get("VERB", 0) + pos_counter.get("AUX", 0)
    adjective_count = pos_counter.get("ADJ", 0)
    adverb_count = pos_counter.get("ADV", 0)
    pronoun_count = pos_counter.get("PRON", 0)

    return StyleMetrics(
        noun_ratio=round(noun_count / total_words, 4),
        verb_ratio=round(verb_count / total_words, 4),
        adjective_ratio=round(adjective_count / total_words, 4),
        adverb_ratio=round(adverb_count / total_words, 4),
        pronoun_ratio=round(pronoun_count / total_words, 4),
        adverbs_manner_count=adverbs_manner_count,
        pos_distribution=dict(pos_counter),
    )
