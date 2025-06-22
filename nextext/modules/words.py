import logging
from collections import Counter, defaultdict
from pathlib import Path

import arabic_reshaper
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import spacy
from bidi.algorithm import get_display
from camel_tools.tokenizers.word import simple_word_tokenize
from matplotlib.figure import Figure
from pyvis.network import Network
from spacy.language import Language
from spacy.tokens import Doc
from wordcloud import WordCloud

from nextext.utils import load_mappings


class WordCounter:
    """
    WordCounter analyzes word frequencies, extracts linguistic features, and generates visualizations from input text.

    Attributes:
        text (str): The text to analyze.
        language (str): The language code of the text.
        font_path (Path): Path to the font used for word cloud rendering.
        nlp (Language | None): Loaded spaCy language model.
        doc (spacy.tokens.Doc | None): spaCy Doc object for the text.
        tokenized_doc (list[str] | None): List of lemmatized, filtered tokens.
        tokenized_nouns (list[str] | None): List of lemmatized nouns from the text.
        word_counts (collections.Counter | None): Counter of word frequencies.
        spacy_entities (dict): Mapping of entity types for NER.
        noun_df (pd.DataFrame | None): DataFrame of high-frequency nouns with associated verbs and adjectives.

    Methods:
        __init__(text: str, language: str, spacy_models_file: str = "spacy_models.json", spacy_entities_file: str = "spacy_entities.json", font_file: str = "Amiri-Regular.ttf") -> None:
            Initializes the WordCounter object with text, language, and configuration files.
        _create_absolute_path(file: str, path: Path = Path("utils")) -> Path:
            Returns an absolute path for a given file and directory.
        _load_spacy_model(spacy_languages: dict[str, str], language: str) -> Language | None:
            Loads the spaCy model for the specified language code.
        text_to_doc() -> None:
            Converts the input text to a spaCy Doc object.
        lemmatize_doc() -> None:
            Tokenizes and lemmatizes the text, populating tokenized_doc and tokenized_nouns.
        count_words(n_words: int = 30, columns: list[str] = ["Word", "Frequency"]) -> pd.DataFrame:
            Counts word frequencies and returns a DataFrame of the most common words.
        named_entity_recognition(columns: list[str] = ["Category", "Entity", "Frequency"]) -> pd.DataFrame:
            Performs named entity recognition on the text and returns a DataFrame.
        get_noun_adjectives(n_freq_nouns: int = 50, n_freq_adjs: int = 5, columns: list[str] = ["Noun", "Adjective"]) -> pd.DataFrame:
            Finds the most frequent nouns and their associated adjectives, returning a DataFrame.
        construct_noun_sentiment_graph(columns: list[str] = ["Noun", "Verb", "Adjective"]) -> nx.Graph:
            Constructs a semantic graph of nouns, verbs, and adjectives from the noun sentiment DataFrame.
        create_interactive_graph() -> str:
            Exports the noun-verb-adjective graph as an interactive HTML file.
        create_wordcloud() -> Figure:
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
        spacy_languages = load_mappings(spacy_models_file)
        self.spacy_entities = load_mappings(spacy_entities_file)
        # Path to the font used for word cloud rendering
        self.font_path = self._create_absolute_path(font_file)

        self.nlp = self._load_spacy_model(spacy_languages, language)
        self.doc: Doc | None = None
        self.tokenized_doc: list | None = None
        self.tokenized_nouns: list | None = None
        self.word_counts: Counter | None = None
        self.noun_df: pd.DataFrame | None = None

    def _create_absolute_path(
        self, file: str, path: Path = Path("utils") / "fonts"
    ) -> Path:
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
    ) -> Language | None:
        """
        Load the spaCy model for the specified language.

        Args:
            spacy_languages (dict[str, str]): Mapping of language codes to spaCy model names.
            language (str): The language code for the text.

        Returns:
            Language | None: The loaded spaCy model or None if loading fails.
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

    def count_words(
        self, n_words: int = 30, columns: list[str] = ["Word", "Frequency"]
    ) -> pd.DataFrame:
        """
        Perform n-gram analysis and count word frequencies using Counter.

        Args:
            n_words (int): Number of top words to return. Defaults to 30.
            columns (list[str]): Column names for the resulting DataFrame. Defaults to ["Word", "Frequency"].

        Returns:
            pd.DataFrame: DataFrame of words and their counts, sorted by frequency.
        """
        try:
            if self.tokenized_doc is None:
                self.logger.error(
                    "Tokenized document is None. Please run lemmatize_doc() first."
                )
                return pd.DataFrame(columns=columns).reset_index(drop=True)
            self.word_counts = Counter(token for token in self.tokenized_doc)
            df = (
                pd.DataFrame(self.word_counts.most_common(n_words), columns=columns)
                .sort_values(columns[1], ascending=False)
                .reset_index(drop=True)
            )
            return df
        except Exception as e:
            self.logger.error(f"Error counting word frequencies: {e}", exc_info=True)
            return pd.DataFrame(columns=columns).reset_index(drop=True)

    def named_entity_recognition(
        self, columns: list[str] = ["Category", "Entity", "Frequency"]
    ) -> pd.DataFrame:
        """
        Perform named entity recognition on the text.

        Args:
            columns (list[str]): Column names for the resulting DataFrame. Defaults to ["Category", "Entity", "Frequency"].

        Returns:
            pd.DataFrame: DataFrame containing named entities and their counts.
        """
        try:
            if self.doc is None:
                self.logger.error("spaCy doc is None. Please run text_to_doc() first.")
                return pd.DataFrame(columns=columns).reset_index(drop=True)
            ent_types = set(self.spacy_entities.keys())
            doc_ents = [
                (ent.text, ent.label_)
                for ent in self.doc.ents
                if ent.label_ in ent_types and len(ent.text.strip()) >= 3
            ]
            entities_count = Counter(doc_ents)
            df = pd.DataFrame(
                [
                    (label, text, count)
                    for (text, label), count in entities_count.items()
                ],
                columns=columns,
            ).reset_index(drop=True)
            return df
        except Exception as e:
            self.logger.error(
                f"Error performing named entity recognition: {e}", exc_info=True
            )
            return pd.DataFrame(columns=columns).reset_index(drop=True)

    def get_noun_sentiment(
        self,
        n_freq_nouns: int | None = None,
        n_freq_verbs: int | None = None,
        n_freq_adjs: int | None = None,
        columns: list[str] = ["Noun", "Verb", "Adjective"],
    ) -> pd.DataFrame:
        """
        Retrieve, for each high-frequency noun, its most common governing verbs
        *and* its most common modifying adjectives, returning a single tidy
        DataFrame.

        Args:
            n_freq_nouns (int | None): How many top nouns to keep. Defaults to None (all nouns).
            n_freq_verbs (int | None): How many top verbs to keep per noun. Defaults to None (all verbs).
            n_freq_adjs (int | None): How many top adjectives to keep per noun. Defaults to None (all adjectives).
            columns (list[str]): Column names for the resulting DataFrame.
                                 Defaults to ["Noun", "Verb", "Adjective"].

        Returns:
            pd.DataFrame: One row per noun with two comma-separated columns listing
                          verbs and adjectives.
        """
        try:
            if self.doc is None:
                self.logger.error("spaCy doc is None. Please run text_to_doc() first.")
                return pd.DataFrame(columns=columns).reset_index(drop=True)

            # Identify top‑frequency noun lemmas
            top_nouns = {
                noun
                for noun, _ in Counter(self.tokenized_nouns).most_common(n_freq_nouns)
            }

            noun_verb_map: dict[str, list[str]] = defaultdict(list)
            noun_adj_map: dict[str, list[str]] = defaultdict(list)

            # Single pass through doc to collect verbs *and* adjectives
            for token in self.doc:
                if token.pos_ == "NOUN":
                    noun_lemma = token.lemma_.lower()
                    if noun_lemma not in top_nouns:
                        continue

                    # Governing verb (head) ---------------------------------
                    if token.head.pos_ == "VERB":
                        noun_verb_map[noun_lemma].append(token.head.lemma_.lower())

                    # Adjectival modifiers ---------------------------------
                    for child in token.children:
                        if child.pos_ == "ADJ":
                            noun_adj_map[noun_lemma].append(child.lemma_.lower())

                # Also capture pattern where adjective precedes noun (amod)
                elif token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
                    noun_lemma = token.head.lemma_.lower()
                    if noun_lemma in top_nouns:
                        noun_adj_map[noun_lemma].append(token.lemma_.lower())

            # Build output rows --------------------------------------------
            rows = []
            for noun in sorted(top_nouns):
                verbs = noun_verb_map.get(noun, [])
                adjs = noun_adj_map.get(noun, [])

                top_verbs = [v for v, _ in Counter(verbs).most_common(n_freq_verbs)]
                top_adjs = [a for a, _ in Counter(adjs).most_common(n_freq_adjs)]

                rows.append(
                    {
                        columns[0]: noun,
                        columns[1]: ", ".join(top_verbs),
                        columns[2]: ", ".join(top_adjs),
                    }
                )

            self.noun_df = pd.DataFrame(rows, columns=columns).reset_index(drop=True)
            return self.noun_df

        except Exception as e:
            self.logger.error(
                f"Error combining noun‑verb‑adjective extraction: {e}", exc_info=True
            )
            return pd.DataFrame(columns=columns).reset_index(drop=True)

    def construct_noun_sentiment_graph(
        self,
        columns: list[str] = ["Noun", "Verb", "Adjective"],
    ) -> nx.Graph:
        """
        Build a graph from the noun-verb-adjective DataFrame.

        Args:
            columns (list[str]): Column names for the DataFrame. Defaults to ["Noun", "Verb", "Adjective"].

        Returns:
            nx.Graph: The constructed semantic graph.
        """
        try:
            if self.noun_df is None:
                self.logger.error(
                    "Noun DataFrame is None. Please run get_noun_adjectives() first."
                )
                return nx.Graph()

            G = nx.Graph()
            for _, row in self.noun_df.iterrows():
                noun = row[columns[0]]
                verbs = row[columns[1]].split(", ") if row[columns[1]] else []
                adjs = row[columns[2]].split(", ") if row[columns[2]] else []

                G.add_node(noun, type="noun")

                for v in verbs:
                    G.add_node(v, type="verb")
                    G.add_edge(noun, v, relation="verb")

                for a in adjs:
                    G.add_node(a, type="adj")
                    G.add_edge(noun, a, relation="adj")
            return G
        except Exception as e:
            self.logger.error(
                f"Error building noun-verb-adjective graph: {e}", exc_info=True
            )
            return nx.Graph()

    def create_interactive_graph(self) -> str:
        """
        Export the noun-verb-adjective graph as an interactive HTML file.

        Returns:
            str: HTML string of the interactive graph.
        """
        try:
            G = self.construct_noun_sentiment_graph()
            if G.number_of_nodes() == 0:
                self.logger.warning("Graph is empty. No nodes to visualize.")
                return "<p>No data to visualize.</p>"

            net = Network(notebook=False, bgcolor="#000000", font_color="white")
            for node, attr in G.nodes(data=True):
                net.add_node(
                    node,
                    label=node,
                    color={"noun": "#4f81bd", "verb": "#9bbb59", "adj": "#c0504d"}.get(
                        attr["type"], "#dddddd"
                    ),
                )

            for u, v, d in G.edges(data=True):
                net.add_edge(u, v, title=d["relation"])

            # Remove annoying white padding around the graph (more precise)
            html = net.generate_html()
            html = html.replace(
                "<body>",
                """<body style="margin:0;padding:0;background-color:#222222;height:100vh;">""",
            ).replace(
                '<div id="mynetwork"',
                '<div id="mynetwork" style="height:100vh;width:100%;background-color:#222222;border:none;"',
            )
            return html
        except Exception as e:
            self.logger.error(f"Error exporting interactive graph: {e}", exc_info=True)
            return "<p>Error generating graph visualization.</p>"

    def create_wordcloud(self) -> Figure:
        """
        Create a wordcloud of the most frequent words.

        Returns:
            Figure: A matplotlib figure containing a wordcloud of the most frequent words.
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
            # Return an empty black figure if wordcloud creation fails
            fig = plt.figure(figsize=(20, 10), facecolor="k")
            plt.axis("off")
            plt.tight_layout(pad=0)
            return fig
