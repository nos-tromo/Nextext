"""Word-level analysis: counts and word clouds via spaCy + NLTK."""

from collections import Counter

import arabic_reshaper
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from bidi.algorithm import get_display
from camel_tools.tokenizers.word import simple_word_tokenize
from dotenv import load_dotenv
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

load_dotenv()


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
