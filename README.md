# prose-metrics

A static, deterministic analyzer for literary fiction and plain prose.

`prose-metrics` is a minimalist Python package that extracts statistics on style, rhythm, vocabulary, dialogue, and readability from raw text. Built on [spaCy](https://spacy.io/) and [textstat](https://github.com/textstat/textstat), it enables technical, deterministic evaluation of prose without relying on AI or network calls at analysis time.

## Key characteristics

- **Agnostic input:** accepts plain text strings only. Cleaning Markdown/HTML is the responsibility of the calling tool.
- **Zero I/O:** the package never reads files from disk; it analyzes only `str` values passed in memory.
- **Deterministic:** the same input text and settings always produce the same report.
- **Minimalist data model:** results are returned as native frozen `dataclasses` (`slots=True`) — no Pydantic or other infrastructure dependencies.
- **Bilingual focus:** optimized for English (`en`) and French (`fr`) texts.

## Requirements

- Python **3.12 – 3.13**
- Runtime dependencies (installed automatically): `spacy`, `textstat`
- Default spaCy language models (installed separately):
  - English: `en_core_web_sm`
  - French: `fr_core_news_sm`

## Installation

This package is available on [PyPI](https://pypi.org/project/prose-metrics/). Install it with:

```bash
# Install the package into your environment
pip install prose-metrics
```

Install the spaCy language model(s) you need:

```bash
uv run python -m spacy download en_core_web_sm
uv run python -m spacy download fr_core_news_sm
```

## Quick start

The simplest way to analyze a text is the top-level `analyze()` function:

```python
from prose_metrics import analyze

text = (
    "The night was cold. She walked quickly through the empty street, "
    "wondering whether anyone had noticed. \"Who is there?\" she asked."
)

report = analyze(text, language="en")

print(report.volume.word_count)
print(report.rhythm.avg_sentence_length)
print(report.readability.flesch_reading_ease)
print(report.to_dict())  # full report as a plain dictionary
```

For repeated analysis, use a `TextAnalyzer` instance to fix the language/model configuration once; the underlying spaCy pipeline is cached by a thread-safe singleton manager:

```python
from prose_metrics import TextAnalyzer

analyzer = TextAnalyzer(language="fr")
report = analyzer.analyze("« Bonjour », dit-il. Il parlait doucement.")
```

### Selecting specific metrics

By default all metrics are computed. You can restrict the analysis to a subset:

```python
report = analyze(text, language="en", metrics=["volume", "readability"])
```

Available metric names: `"volume"`, `"rhythm"`, `"style"`, `"vocabulary"`, `"readability"`. Use `"all"` (default) or an empty sequence to compute everything. Metrics not requested are `None` in the resulting `TextReport`.

### Tunable parameters

| Parameter | Default | Description |
|---|---|---|
| `language` | `"en"` | ISO 639-1 language code; supported pipeline defaults: `en`, `fr`. |
| `model_name` | `None` | Explicit spaCy model name, overriding the language default. |
| `doc` | `None` | Optional pre-parsed spaCy `Doc` to bypass re-tokenization. |
| `mattr_window_size` | `100` | Sliding-window size for the Moving Average Type-Token Ratio. |
| `words_per_minute` | `200` | Reading speed used for the estimated reading time. |

## The report structure

`analyze()` returns a `TextReport` dataclass with:

- `language` — language code used for the analysis
- `execution_time_seconds` — wall-clock processing time
- `volume` — `VolumeMetrics`
- `rhythm` — `RhythmMetrics`
- `style` — `StyleMetrics`
- `vocabulary` — `VocabularyMetrics`
- `readability` — `ReadabilityMetrics`

Call `report.to_dict()` to obtain a fully nested dictionary representation.

### Volume (`VolumeMetrics`)

Character/word/sentence/paragraph counts and dialogue statistics:

- `character_count`, `character_count_no_spaces`
- `word_count`, `sentence_count`, `paragraph_count`
- `dialogue_word_count`, `narrative_word_count`, `dialogue_ratio`

Dialogue detection recognizes curly quotes `“...”`, French guillemets `« ... »`, straight double quotes `"..."`, and leading dashes (`—`, `–`, `-`) at line start.

### Rhythm (`RhythmMetrics`)

Sentence-length dispersion and punctuation cadence:

- `avg_sentence_length`, `sentence_length_variance`, `sentence_length_std_dev`
- `short_sentence_ratio` (sentences with fewer than 10 words)
- `long_sentence_ratio` (sentences with more than 30 words)
- `punctuation_distribution` — counts for `, . ! ? ; : — …`

### Style (`StyleMetrics`)

Grammatical composition based on spaCy universal POS tags:

- `noun_ratio`, `verb_ratio`, `adjective_ratio`, `adverb_ratio`, `pronoun_ratio`
  (noun counts include proper nouns; verb counts include auxiliaries)
- `adverbs_manner_count` — manner adverbs detected via syntactic dependency checks and suffix rules (`-ly` in English, `-ment` in French), with curated irregular and exclusion lists
- `pos_distribution` — raw counts per universal POS tag

### Vocabulary (`VocabularyMetrics`)

Lexical richness indicators:

- `unique_word_count` — distinct lemmas
- `ttr` — Type-Token Ratio
- `mattr` — Moving Average Type-Token Ratio (with `mattr_window_size`)
- `hapax_count`, `hapax_ratio` — words occurring exactly once

### Readability (`ReadabilityMetrics`)

- `flesch_reading_ease`, `flesch_kincaid_grade`, `gunning_fog`
- `estimated_reading_time_minutes`

Note: readability scores are computed via `textstat`; the Gunning Fog index is reported as `0.0` for non-English texts because it is only supported for English by `textstat`. Readability is supported for `en`, `es`, `fr`, `it`, `de`, `nl`.

## Architecture notes

- `prose_metrics.analyzer` — orchestrates parsing and metric computation (`TextAnalyzer`, `analyze`).
- `prose_metrics.metrics` — independent computation functions for volume, rhythm, style, vocabulary, and readability.
- `prose_metrics.models.report` — frozen, slotted dataclasses for all report models.
- `prose_metrics.nlp.pipeline` — a thread-safe singleton `SpacyPipelineManager` that caches spaCy pipelines and disables unnecessary components (by default, `ner`) for performance.
- `prose_metrics.nlp.exceptions` — package-specific exceptions.

## License

MIT — see [LICENSE](LICENSE) for details.
