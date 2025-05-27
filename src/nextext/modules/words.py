import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import arabic_reshaper
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from bidi.algorithm import get_display
from camel_tools.tokenizers.word import simple_word_tokenize
from spacy.tokens import Doc
from wordcloud import WordCloud

from nextext.utils import load_lang_maps


class WordCounter:
    """
    WordCounter analyzes word frequencies, extracts linguistic features, and generates visualizations from input text.

    Attributes:
        text (str): The text to analyze.
        language (str): The language code of the text.
        font_path (Path): Path to the font used for word cloud rendering.
        nlp (spacy.language.Language | None): Loaded spaCy language model.
        doc (spacy.tokens.Doc | None): spaCy Doc object for the text.
        tokenized_doc (list[str] | None): List of lemmatized, filtered tokens.
        tokenized_nouns (list[str] | None): List of lemmatized nouns from the text.
        word_counts (collections.Counter | None): Counter of word frequencies.
        spacy_entities (dict): Mapping of entity types for NER.

    Methods:
        _create_absolute_path(file: str, path: Path = Path("utils")) -> Path:
            Returns an absolute path for a given file and directory.
        _load_spacy_model(spacy_languages: dict[str, str], language: str) -> spacy.language.Language | None:
            Loads the spaCy model for the specified language code.
        text_to_doc() -> None:
            Converts the input text to a spaCy Doc object.
        lemmatize_doc() -> None:
            Tokenizes and lemmatizes the text, populating tokenized_doc and tokenized_nouns.
        count_words(n_words: Optional[int] = None) -> list[tuple[str, int]]:
            Counts word frequencies and returns the most common words.
        named_entity_recognition() -> list[tuple[tuple[str, str], int]] | None:
            Performs named entity recognition on the text.
        get_noun_adjectives(n_freq_nouns: int = 50, n_freq_adjs: int = 5) -> dict[str, list[str]]:
            Finds the most frequent nouns and their associated adjectives.
        create_wordcloud() -> plt.Figure:
            Generates a word cloud visualization of word frequencies.
    """

    def __init__(
        self,
        text: str,
        language: str,
        spacy_models_file: str = "spacy_models.json",
        spacy_entities_file: str = "spacy_entities.json",
        font_file: str = "Amiri-Regular.ttf",
    ) -> None:
        """
        Initialize the WordCount object.

        Args:
            text (str): The transcript text to analyse.
            language (str): The language of the text.
            spacy_language_path (str): Path to the JSON file containing spaCy language models. Defaults to "utils/spacy_languages.json".
            spacy_entities_path (str): Path to the JSON file containing spaCy entity types. Defaults to "utils/spacy_entities.json".
            font_file (str): Path to the font used to write the word cloud. Defaults to "static/fonts/Amiri-Regular.ttf".
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.text = text
        self.language = language

        # Paths to the JSON files containing spaCy language models and entity types
        spacy_languages, _ = load_lang_maps(spacy_models_file)
        self.spacy_entities, _ = load_lang_maps(spacy_entities_file)
        # Path to the font used for word cloud rendering
        self.font_path = self._create_absolute_path(font_file)

        self.nlp = self._load_spacy_model(spacy_languages, language)
        self.doc: Optional[spacy.tokens.Doc] = None
        self.tokenized_doc: list | None = None
        self.tokenized_nouns: list | None = None
        self.word_counts: Counter | None = None

    def _create_absolute_path(self, file: str, path: Path = Path("utils")) -> Path:
        """
        Create an absolute path from a relative path.

        Args:
            path (Path): The relative or absolute path to be converted.

        Returns:
            Path: The absolute path.
        """
        root = Path(__file__).resolve().parent.parent
        return path if path.is_absolute() else root / path / file

    def _load_spacy_model(
        self, spacy_languages: dict[str, str], language: str
    ) -> spacy.language.Language | None:
        """
        Load the spaCy model for the specified language.

        Args:
            spacy_languages (dict[str, str]): Mapping of language codes to spaCy model names.
            language (str): The language code for the text.

        Returns:
            spacy.language.Language | None: The loaded spaCy model or None if loading fails.
        """
        try:
            if language == "ar":
                nlp = spacy.blank("ar")

                def camel_tokenizer(text: str) -> Doc:
                    words = simple_word_tokenize(text)
                    return Doc(nlp.vocab, words=words)

                nlp.tokenizer = camel_tokenizer
                return nlp

            model_name = spacy_languages.get(language, "xx")
            try:
                return spacy.load(model_name)
            except Exception as e:
                self.logger.warning(
                    f"Primary spaCy model '{model_name}' failed. Falling back to 'xx_sent_ud_sm': {e}"
                )
                return spacy.load("xx_sent_ud_sm")

        except Exception as e:
            self.logger.error(
                f"Failed to load any language model for language '{language}': {e}"
            )
            return None

    def text_to_doc(self) -> None:
        """
        Convert the text to a spaCy doc object.
        """
        try:
            if self.nlp is not None:
                self.doc = self.nlp(self.text)
            else:
                self.logger.error(
                    "spaCy language model is not loaded. Cannot process text."
                )
                self.doc = None
        except Exception as e:
            self.logger.error(f"Unable to convert text to spaCy doc: {e}")

    def lemmatize_doc(self) -> None:
        """
        Tokenize and lemmatize the text.

        Returns:
            list[str]: List of tokenized and lemmatized words.
        """
        try:
            if self.doc is None:
                self.logger.error("spaCy doc is None. Please run text_to_doc() first.")
                self.tokenized_doc = []
                self.tokenized_nouns = []
                return

            if self.language == "ar":
                # No lemmatization or POS tagging for Arabic
                self.tokenized_doc = [
                    token.text for token in self.doc if token.is_alpha
                ]
                self.tokenized_nouns = []  # Can't extract nouns without POS
            else:
                self.tokenized_doc = [
                    token.lemma_.lower()
                    for token in self.doc
                    if token.is_alpha and not token.is_stop
                ]
                self.tokenized_nouns = [
                    token
                    for token in self.tokenized_doc
                    if any(
                        t.lemma_.lower() == token and t.pos_ == "NOUN" for t in self.doc
                    )
                ]
        except Exception as e:
            self.logger.error(f"Error tokenizing words: {e}", exc_info=True)
            raise

    def count_words(self, n_words: Optional[int] = None) -> pd.DataFrame:
        """
        Perform n-gram analysis and count word frequencies using Counter.

        Args:
            n_words (int, optional): Number of top words to return. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame of words and their counts, sorted by frequency.
        """
        try:
            if self.tokenized_doc is None:
                self.logger.error(
                    "Tokenized document is None. Please run lemmatize_doc() first."
                )
                return pd.DataFrame(columns=["word", "count"]).set_index("word")
            self.word_counts = Counter(token for token in self.tokenized_doc)
            return (
                pd.DataFrame(
                    self.word_counts.most_common(n_words), columns=["word", "count"]
                )
                .sort_values("count", ascending=False)
                .set_index("word")
            )
        except Exception as e:
            self.logger.error(f"Error counting word frequencies: {e}", exc_info=True)
            return pd.DataFrame(columns=["word", "count"]).set_index("word")

    def named_entity_recognition(self) -> list[tuple[tuple[str, str], int]] | None:
        """
        Perform named entity recognition on the text.

        Returns:
            list[tuple[tuple[str, str], int]]: List of tuples containing named entities and their counts.
        """
        try:
            if self.doc is None:
                self.logger.error("spaCy doc is None. Please run text_to_doc() first.")
                return None
            ent_types = set(self.spacy_entities.keys())
            doc_ents = [
                (ent.text, ent.label_)
                for ent in self.doc.ents
                if ent.label_ in ent_types and len(ent.text.strip()) >= 3
            ]
            entities_count = Counter(doc_ents)
            return pd.DataFrame(
                [
                    (label, text, count)
                    for (text, label), count in entities_count.items()
                ],
                columns=["Category", "Entity", "Frequency"],
            ).reset_index(drop=True)
        except Exception as e:
            self.logger.error(
                f"Error performing named entity recognition: {e}", exc_info=True
            )
            return None

    def get_noun_adjectives(
        self, n_freq_nouns: int = 50, n_freq_adjs: int = 5
    ) -> pd.DataFrame:
        """
        Get the most frequent nouns and their associated adjectives.

        Args:
            n_freq_nouns (int, optional): Number of top nouns to consider. Defaults to 50.
            n_freq_adjs (int, optional): Number of top adjectives to consider. Defaults to 5.

        Returns:
            pd.DataFrame: DataFrame containing nouns and their associated adjectives.
        """
        try:
            if self.doc is None:
                self.logger.error("spaCy doc is None. Please run text_to_doc() first.")
                return pd.DataFrame(columns=["Noun", "Adjective"])

            # Get top nouns
            top_nouns = [
                word
                for word, _ in Counter(self.tokenized_nouns).most_common(n_freq_nouns)
            ]
            noun_adj_map = defaultdict(list)

            # Collect adjectives for each noun
            for token in self.doc:
                if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
                    noun_lemma = token.head.lemma_.lower()
                    if noun_lemma in top_nouns:
                        noun_adj_map[noun_lemma].append(token.lemma_.lower())

            # Prepare DataFrame rows
            data = []
            for noun, adjs in noun_adj_map.items():
                adj_counts = Counter(adjs)
                top_adjs = [adj for adj, _ in adj_counts.most_common(n_freq_adjs)]
                adj_str = ", ".join(top_adjs)
                data.append({"Noun": noun, "Adjective": adj_str})

            return pd.DataFrame(data, columns=["Noun", "Adjective"])
        except Exception as e:
            self.logger.error(f"Error getting noun-adjective pairs: {e}", exc_info=True)
            return pd.DataFrame(columns=["Noun", "Adjective"])

    def create_wordcloud(self) -> plt.Figure:
        """
        Create a wordcloud of the most frequent words.

        Returns:
            plt.Figure: A matplotlib figure containing a wordcloud of the most frequent words.
        """
        try:
            # Create a string of words for the wordcloud
            if self.word_counts is None:
                self.logger.error(
                    "Word counts are not available. Please run count_words() first."
                )
                raise ValueError(
                    "Word counts are not available. Please run count_words() first."
                )
            text = " ".join(
                [
                    str(word)
                    for word, count in self.word_counts.items()
                    for _ in range(count)
                ]
            )
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
        except Exception as e:
            self.logger.error(f"Error creating wordcloud: {e}", exc_info=True)
            raise
