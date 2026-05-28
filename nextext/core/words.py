"""Word-level analysis: counts, GLiNER NER, and word clouds via spaCy + NLTK."""

import hashlib
import json
import os
import re
import shutil
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, cast

import arabic_reshaper
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from bidi.algorithm import get_display
from camel_tools.tokenizers.word import simple_word_tokenize
from dotenv import load_dotenv
from gliner import GLiNER
from loguru import logger
from matplotlib.figure import Figure
from spacy.language import Language
from spacy.tokens import Doc
from wordcloud import WordCloud

from nextext.utils.font_loader import load_font_file
from nextext.utils.mappings_loader import load_mappings
from nextext.utils.model_loader import (
    download_spacy_model,
    ensure_spacy_model_path,
)
from nextext.utils.model_registry import REGISTRY, ModelSpec, Strategy

load_dotenv()

# ---------------------------------------------------------------------------
# GLiNER NER — module-level singleton and offline loading helpers
# ---------------------------------------------------------------------------

_GLINER_MODEL_ID = os.getenv("NER_MODEL", "gliner-community/gliner_large-v2.5")
_GLINER_LABELS = [
    "date",
    "event",
    "fac",
    "group",
    "loc",
    "money",
    "org",
    "person",
    "time",
]
_GLINER_THRESHOLD = 0.3
_GLINER_WORD_BUDGET = 512
_GLINER_OFFLINE_DIR = Path(tempfile.gettempdir()) / "nextext-gliner-offline"
_SENTENCE_RE = re.compile(
    r".+?(?:[.!?]+[\"')\]]*(?=\s+|$)|\n{2,}|$)",
    re.DOTALL,
)


def _resolve_hf_cache_dir() -> Path:
    """Return the Hugging Face hub cache directory.

    Returns:
        Path: Resolved HF hub cache directory from environment or default.
    """
    hf_cache = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if hf_cache:
        return Path(hf_cache)
    return Path.home() / ".cache" / "huggingface" / "hub"


def _resolve_hf_cache_path(cache_dir: Path, repo_id: str) -> Path | None:
    """Find a locally cached HF snapshot directory for repo_id.

    Args:
        cache_dir (Path): HF hub cache directory (e.g.
            ``~/.cache/huggingface/hub``).
        repo_id (str): HuggingFace repository ID (e.g.
            ``"gliner-community/gliner_large-v2.5"``).

    Returns:
        Path | None: Path to the snapshot directory if found, otherwise
            ``None``.
    """
    model_dir_name = f"models--{repo_id.replace('/', '--')}"
    model_cache_dir = cache_dir / model_dir_name
    if not model_cache_dir.exists():
        return None
    ref_path = model_cache_dir / "refs" / "main"
    if not ref_path.exists():
        return None
    commit_hash = ref_path.read_text().strip()
    snapshot_path = model_cache_dir / "snapshots" / commit_hash
    return snapshot_path if snapshot_path.exists() else None


def _load_gliner_config(model_dir: Path) -> dict[str, Any]:
    """Load the GLiNER config for a local model directory.

    Args:
        model_dir (Path): Directory containing ``gliner_config.json``.

    Returns:
        dict[str, Any]: Parsed GLiNER configuration payload.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = model_dir / "gliner_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"No GLiNER config found in {model_dir}")
    return cast(dict[str, Any], json.loads(config_path.read_text(encoding="utf-8")))


def _link_or_copy_path(source: Path, destination: Path) -> None:
    """Materialize a file or directory at ``destination`` from ``source``.

    Args:
        source (Path): Existing file or directory to link or copy.
        destination (Path): Target path that should mirror ``source``.
    """
    if destination.exists():
        return
    try:
        destination.symlink_to(source, target_is_directory=source.is_dir())
    except Exception:
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)


def _resolve_local_gliner_dependency(cache_dir: Path, dependency: str) -> Path:
    """Resolve a GLiNER dependency path without allowing network access.

    Args:
        cache_dir (Path): Hugging Face hub cache directory.
        dependency (str): Repo ID or local filesystem path referenced by
            GLiNER config.

    Returns:
        Path: Local filesystem path for the dependency.

    Raises:
        FileNotFoundError: If the dependency is unavailable locally.
    """
    dep_path = Path(dependency).expanduser()
    if dep_path.exists():
        return dep_path.resolve()
    resolved = _resolve_hf_cache_path(cache_dir=cache_dir, repo_id=dependency)
    if resolved is None:
        raise FileNotFoundError(
            f"GLiNER offline load requires a local snapshot for '{dependency}', but none was found in {cache_dir}."
        )
    return resolved


def _materialize_offline_gliner_dir(model_dir: Path, config: dict[str, Any]) -> Path:
    """Create a local-only GLiNER directory with patched config references.

    Args:
        model_dir (Path): Original local GLiNER model directory.
        config (dict[str, Any]): GLiNER config payload to write into the
            offline runtime directory.

    Returns:
        Path: Local runtime directory with patched config and links to model
            assets.
    """
    digest = hashlib.sha256(f"{model_dir.resolve()}\0{json.dumps(config, sort_keys=True)}".encode()).hexdigest()[:16]
    runtime_dir = _GLINER_OFFLINE_DIR / digest
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for item in model_dir.iterdir():
        if item.name == "gliner_config.json":
            continue
        _link_or_copy_path(item, runtime_dir / item.name)
    (runtime_dir / "gliner_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return runtime_dir


def _prepare_local_gliner_model_dir(model_dir: Path, cache_dir: Path) -> Path:
    """Prepare a GLiNER directory for strict offline loading.

    Rewrites backbone config references (model_name, labels_encoder,
    labels_decoder) to local snapshot paths so the upstream loader does not
    trigger outbound hub resolution.

    Args:
        model_dir (Path): Local GLiNER model directory or snapshot path.
        cache_dir (Path): Hugging Face hub cache directory.

    Returns:
        Path: A local model directory safe to hand to
            ``GLiNER.from_pretrained``.
    """
    config = _load_gliner_config(model_dir)
    patched = False
    for field in ("model_name", "labels_encoder", "labels_decoder"):
        value = config.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        resolved = _resolve_local_gliner_dependency(cache_dir=cache_dir, dependency=value)
        resolved_str = str(resolved)
        if value != resolved_str:
            config[field] = resolved_str
            patched = True
    if not patched:
        return model_dir
    runtime_dir = _materialize_offline_gliner_dir(model_dir=model_dir, config=config)
    logger.info("Prepared offline GLiNER runtime directory: {}", runtime_dir)
    return runtime_dir


def _resolve_gliner_load_target(model_id: str, cache_dir: Path) -> tuple[str, bool]:
    """Resolve the load target for GLiNER without allowing accidental hub access.

    Args:
        model_id (str): GLiNER repo ID or local filesystem path.
        cache_dir (Path): Hugging Face hub cache directory.

    Returns:
        tuple[str, bool]: ``(load_target, local_only)`` suitable for
            ``GLiNER.from_pretrained``.

    Raises:
        FileNotFoundError: If offline mode is enabled and the model is not
            cached.
    """
    local_dir = Path(model_id).expanduser()
    if local_dir.exists():
        prepared = _prepare_local_gliner_model_dir(model_dir=local_dir.resolve(), cache_dir=cache_dir)
        return str(prepared), True

    resolved = _resolve_hf_cache_path(cache_dir=cache_dir, repo_id=model_id)
    if resolved is not None:
        logger.info("Using local GLiNER model path: {}", resolved)
        prepared = _prepare_local_gliner_model_dir(model_dir=resolved, cache_dir=cache_dir)
        return str(prepared), True

    if os.getenv("HF_HUB_OFFLINE", "0") == "1" or os.getenv("NEXTEXT_OFFLINE", "0") == "1":
        raise FileNotFoundError(
            f"GLiNER model '{model_id}' is not available in the local cache "
            f"{cache_dir}. Disable offline mode to download it."
        )

    return model_id, False


def _load_gliner() -> Any:
    """Build the GLiNER model fresh from the local HF cache (registry loader).

    Returns:
        Any: A ``GLiNER`` model instance on CPU.

    Raises:
        FileNotFoundError: If ``NEXTEXT_OFFLINE=1`` and the model is not
            present in the local HF Hub cache.
    """
    cache_dir = _resolve_hf_cache_dir()
    load_target, local_only = _resolve_gliner_load_target(_GLINER_MODEL_ID, cache_dir)
    logger.info(
        "Loading GLiNER model '{}' (local_only={}).",
        _GLINER_MODEL_ID,
        local_only,
    )
    model = GLiNER.from_pretrained(load_target, local_files_only=local_only)
    logger.info("GLiNER model loaded.")
    return model


def _move_gliner(model: Any, device: str) -> Any:
    """Move a GLiNER model to ``device``.

    GLiNER's own ``.to()`` covers the wrapped transformer in recent versions,
    but older releases only move the inner ``.model`` attribute.  The public
    API is tried first; if it raises, the inner module is moved directly.

    Args:
        model (Any): A ``GLiNER`` model instance to move.
        device (str): Target device string, e.g. ``"cuda"`` or ``"cpu"``.

    Returns:
        Any: The same ``GLiNER`` model instance after the move.
    """
    to_method = getattr(model, "to", None)
    if callable(to_method):
        try:
            to_method(device)
            return model
        except (AttributeError, TypeError, NotImplementedError):
            pass
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "to"):
        inner.to(device)
    return model


# GLiNER runs on Apple Silicon MPS without raising, but on realistic
# ~512-word chunks the backend silently returns zero predictions while
# the CPU path returns the correct entities (observed with
# gliner_large-v2.5 on torch 2.8, 2026-04-19). A silent wrong answer is
# worse than a crash, so pin to CPU on Mac; CUDA is still used when
# available.
REGISTRY.register(
    ModelSpec(
        name="gliner",
        loader=_load_gliner,
        mover=_move_gliner,
        default_strategy=Strategy.OFFLOAD,
        mps_compatible=False,
    )
)


def _chunk_text(text: str, word_budget: int = _GLINER_WORD_BUDGET) -> list[str]:
    """Split text into sentence-packed chunks within a word budget.

    Args:
        text (str): Raw input text.
        word_budget (int): Maximum whitespace-delimited words per chunk.

    Returns:
        list[str]: Ordered list of text chunks suitable for repeated GLiNER
            inference.
    """
    sentences = [m.group(0).strip() for m in _SENTENCE_RE.finditer(text.strip()) if m.group(0).strip()] or [
        text.strip()
    ]
    chunks: list[str] = []
    current_words: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if len(current_words) + len(words) > word_budget:
            if current_words:
                chunks.append(" ".join(current_words))
            current_words = words[:word_budget]
        else:
            current_words.extend(words)
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


class WordCounter:
    """Analyzes word frequencies, extracts linguistic features, and renders visualizations.

    Attributes:
        text (str): The text to analyze.
        language (str): The language code of the text.
        font_path (Path): Path to the font used for word cloud rendering.
        nlp (Language | None): Loaded spaCy language model.
        doc (spacy.tokens.Doc | None): spaCy Doc object for the text.
        tokenized_doc (list[str] | None): List of lemmatized, filtered tokens.
        tokenized_nouns (list[str] | None): List of lemmatized nouns from the text.
        word_counts (collections.Counter | None): Counter of word frequencies.
        noun_df (pd.DataFrame | None): DataFrame of high-frequency nouns with associated verbs and adjectives.

    Methods:
        __init__(text, language, spacy_models_file=..., font_file=...) -> None:
            Initialise with text, language, and configuration files.
        _load_spacy_model(spacy_languages, language) -> Language | None:
            Load the spaCy model for the specified language code.
        text_to_doc() -> None:
            Convert the input text to a spaCy Doc object.
        lemmatize_doc() -> None:
            Tokenize/lemmatize the text into ``tokenized_doc`` + ``tokenized_nouns``.
        count_words(n_words=30, columns=...) -> pd.DataFrame:
            Count word frequencies; return a top-N DataFrame.
        named_entity_recognition(columns=...) -> pd.DataFrame:
            Run NER and return a DataFrame.
        create_wordcloud() -> Figure:
            Render a word-cloud visualisation of word frequencies.
    """

    def __init__(
        self,
        text: str,
        language: str,
        spacy_models_file: str = "spacy_models.json",
        font_file: str = "Amiri-Regular.ttf",
    ) -> None:
        """Initializes the WordCounter with text, language, and configuration files.

        Args:
            text (str): The text to analyze.
            language (str): Language code of the text (e.g., "en", "ar").
            spacy_models_file (str, optional): JSON file with spaCy model mappings. Defaults to
                "spacy_models.json".
            font_file (str, optional): Font file for word-cloud rendering. Defaults to
                "Amiri-Regular.ttf".
        """
        self.text = text
        self.language = language

        # Load spaCy models
        spacy_languages = load_mappings(spacy_models_file)
        self.nlp = self._load_spacy_model(language, spacy_languages)
        logger.info("Loaded spaCy model for language '{}': {}", language, self.nlp)

        # Set the font path for word cloud generation
        self.font_path = load_font_file(font_file)

        self.doc: Doc | None = None
        self.tokenized_doc: list[str] | None = None
        self.tokenized_nouns: list[str] | None = None
        self.word_counts: Counter[str] | None = None
        self.noun_df: pd.DataFrame | None = None

    def _load_spacy_model(self, language: str, spacy_languages: dict[str, str]) -> Language | None:
        """Load the spaCy model for the specified language code.

        Args:
            language (str): Language code for which to load the spaCy model.
            spacy_languages (dict[str, str]): Mapping of language codes to spaCy model names.

        Returns:
            Language | None: Loaded spaCy model or None if loading fails.
        """
        if language == "ar":
            nlp = spacy.blank("ar")

            def arabic_tokenizer(text: str) -> Doc:
                return Doc(nlp.vocab, words=simple_word_tokenize(text))

            nlp.tokenizer = arabic_tokenizer
            return nlp
        # Add other language-specific handling if needed
        model_id = None
        if language in spacy_languages.keys():
            model_id = spacy_languages.get(language)
        if model_id is None:
            logger.warning(
                "Language '{}' not found in spaCy mappings. Using multilingual model.",
                language,
            )
            model_id = spacy_languages.get("xx")
        if model_id is not None:
            ensure_spacy_model_path()
            download_spacy_model(model_id)
            return spacy.load(model_id)
        else:
            logger.error(
                "No valid spaCy model id found for language '{}'.",
                language,
            )
            return None

    def text_to_doc(self) -> None:
        """Convert the text to a spaCy doc object."""
        if self.nlp is not None:
            self.doc = self.nlp(self.text)
        else:
            logger.error("spaCy language model is not loaded. Cannot process text.")
            self.doc = None

    def lemmatize_doc(self) -> None:
        """Tokenize and lemmatize the text.

        Populates ``self.tokenized_doc`` and ``self.tokenized_nouns`` as a
        side effect; returns nothing.
        """
        if self.doc is None:
            logger.error("spaCy doc is None. Please run text_to_doc() first.")
            self.tokenized_doc = []
            self.tokenized_nouns = []
            return

        if self.language == "ar":
            # No lemmatization or POS tagging for Arabic
            self.tokenized_doc = [token.text for token in self.doc if token.is_alpha]
            self.tokenized_nouns = []  # Can't extract nouns without POS
        else:
            self.tokenized_doc = [token.lemma_.lower() for token in self.doc if token.is_alpha and not token.is_stop]
            self.tokenized_nouns = [
                token
                for token in self.tokenized_doc
                if any(t.lemma_.lower() == token and t.pos_ == "NOUN" for t in self.doc)
            ]

    def count_words(self, n_words: int = 30, columns: list[str] | None = None) -> pd.DataFrame:
        """Perform n-gram analysis and count word frequencies using Counter.

        Args:
            n_words (int): Number of top words to return. Defaults to 30.
            columns (list[str]): Column names for the resulting DataFrame. Defaults to ["Word", "Frequency"].

        Returns:
            pd.DataFrame: DataFrame of words and their counts, sorted by frequency.
        """
        if columns is None:
            columns = ["Word", "Frequency"]
        if self.tokenized_doc is None:
            logger.error("Tokenized document is None. Please run lemmatize_doc() first.")
            return pd.DataFrame(columns=pd.Index(columns)).reset_index(drop=True)
        self.word_counts = Counter(token for token in self.tokenized_doc)
        df = (
            pd.DataFrame(self.word_counts.most_common(n_words), columns=pd.Index(columns))
            .sort_values(columns[1], ascending=False)
            .reset_index(drop=True)
        )
        return df

    def named_entity_recognition(self, columns: list[str] | None = None) -> pd.DataFrame:
        """Perform named entity recognition on the text using GLiNER.

        Args:
            columns (list[str]): Column names for the resulting DataFrame.
                Defaults to ["Category", "Entity", "Frequency"].

        Returns:
            pd.DataFrame: DataFrame containing named entities and their counts.
        """
        if columns is None:
            columns = ["Category", "Entity", "Frequency"]
        if not self.text or not self.text.strip():
            logger.error("Text is empty. Cannot run NER.")
            return pd.DataFrame(columns=pd.Index(columns)).reset_index(drop=True)

        all_entities: list[tuple[str, str]] = []
        try:
            with REGISTRY.acquire("gliner") as model:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*truncat.*max_length.*no maximum length.*",
                    )
                    for chunk in _chunk_text(self.text):
                        try:
                            preds = model.predict_entities(chunk, _GLINER_LABELS, threshold=_GLINER_THRESHOLD)
                            for pred in preds:
                                text_val = pred.get("text", "").strip()
                                label = pred.get("label", "").strip()
                                if text_val and label and len(text_val) >= 3:
                                    all_entities.append((label.upper(), text_val))
                        except Exception as exc:
                            logger.warning("GLiNER chunk inference failed: {}", exc)
        except Exception as exc:
            logger.error("Failed to load GLiNER model: {}", exc)
            return pd.DataFrame(columns=pd.Index(columns)).reset_index(drop=True)

        if not all_entities:
            return pd.DataFrame(columns=pd.Index(columns)).reset_index(drop=True)

        entity_counts = Counter(all_entities)
        return pd.DataFrame(
            [(label, text, count) for (label, text), count in entity_counts.items()],
            columns=pd.Index(columns),
        ).reset_index(drop=True)

    def create_wordcloud(self) -> Figure | None:
        """Create a wordcloud of the most frequent words.

        Returns:
            Figure | None: A matplotlib figure containing a wordcloud of the
            most frequent words, or ``None`` when there are no word counts to
            plot (e.g. very short transcripts that consist only of stopwords).

        Raises:
            ValueError: If ``count_words()`` has not been run yet.
        """
        # Create a string of words for the wordcloud
        if self.word_counts is None:
            logger.error("Word counts are not available. Please run count_words() first.")
            raise ValueError("Word counts are not available. Please run count_words() first.")
        if not self.word_counts:
            logger.info("Word counts are empty; skipping word cloud generation.")
            return None
        text = " ".join([str(word) for word, count in self.word_counts.items() for _ in range(count)])
        if self.language == "ar":
            text = arabic_reshaper.reshape(text)
            text = get_display(text)

        # Generate the word cloud
        wc = WordCloud(
            font_path=self.font_path if self.language == "ar" else None,
            background_color="black",
            width=2000,
            height=1000,
            collocations=False,
        ).generate(text)

        fig = plt.figure(figsize=(20, 10), facecolor="k")
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        return fig
