import logging

import pandas as pd
import pycountry
import torch
from bertopic import BERTopic
from bertopic.representation import PartOfSpeech
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from .ollama_cfg import (
    call_ollama,
    topic_summaries_prompt,
    topic_titles_prompt,
)
from nextext.utils import load_lang_maps


class TopicModeling:
    """
    TopicModeling is a class for performing topic modeling on text data using the BERTopic library.
    It includes methods for loading models, fitting the topic model, and visualizing the results.
    The class also handles language-specific processing using spaCy and supports zero-shot topic modeling.
    The class is designed to be flexible and can be adapted for different languages and models.

    Attributes:
        rows (list[str]): List of text rows to be processed.
        language (str): Language code for the text data.
        sentence_model (str): Sentence transformer model name.
        device (str): Device for computation ('cuda', 'mps', or 'cpu').
        n_min_clusters (int): Minimum number of clusters for HDBSCAN.
        stop_words (list[str]): List of stop words for the specified language.
        topic_model (BERTopic | None): BERTopic model instance.
        nlp (spacy.language.Language | None): spaCy language model for the specified language.
        zeroshot_topics (list[str] | None): List of zero-shot topics.
        logger (logging.Logger): Logger instance for logging messages.

    Methods:
        _convert_to_language_name(language: str) -> str | None:
            Converts the language code to its language name to be handled by the BERTopic object.
        _load_spacy_model(language: str) -> str | None:
            Loads the spaCy language model based on the provided language code.
        load_umap_model(n_neighbors: int = 15, n_components: int = 5, min_dist: float = 0.0, metric: str = "cosine", random_state: int = 42) -> UMAP | None:
            Loads the UMAP model with the specified parameters.
        load_vectorizer_model(ngram_range: tuple = (1, 2), min_df: int = 2) -> CountVectorizer | None:
            Loads the CountVectorizer model with the specified parameters.
        load_representation_model() -> PartOfSpeech | None:
            Loads the representation model for BERTopic.
        load_pipeline(top_n_words: int = 10, calculate_probabilities: bool = True, verbose: bool = True, nr_topics: str = "auto") -> None:
            Loads the complete pipeline for topic modeling.
        fit_model() -> pd.DataFrame | None:
            Fits the topic model to the data.
        summarize_topics(language: str = "German") -> list[tuple[str, str]] | None:
            Summarizes the topics using zero-shot learning.
    """

    def __init__(
        self,
        rows: list[str],
        lang_code: str,
        spacy_language_file: str = "spacy_models.json",
        sentence_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    ) -> None:
        """
        Initialize the TopicModeling object.

        Args:
            rows (list[str]): List of text rows to be processed.
            language (str): Language code for the text data.
            spacy_language_path (Path): Path to the JSON file containing spaCy language models. Defaults to "utils/spacy_languages.json".
            sentence_model (str, optional): Sentence transformer model name. Defaults to "thenlper/gte-small".
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.docs = rows
        spacy_languages, _ = load_lang_maps(spacy_language_file)
        self.language = self._convert_to_language_name(lang_code)
        self.nlp_name = self._load_spacy_model(spacy_languages, lang_code)
        self.logger.info(
            f"spaCy model '{self.nlp_name}' loaded for language '{lang_code}'."
        )

        self.sentence_model = sentence_model
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.stop_words = stopwords.words(self.language, "english")
        self.topic_model: BERTopic | None = None
        self.topic_df: pd.DataFrame | None = None

    def _convert_to_language_name(self, language: str) -> str | None:
        """
        Converts the language code to its language name to be handled by the BERTopic object.

        Args:
            language (str): Language code to be converted.

        Returns:
            str | None: The language name as a string, or None if the conversion fails.
        """
        try:
            return pycountry.languages.get(alpha_2=language.lower()).name.lower()
        except Exception as e:
            self.logger.error(f"Error converting language: {e}.", exc_info=True)
            return None

    def _load_spacy_model(
        self, spacy_languages: dict[str, str], language: str
    ) -> str | None:
        """
        Load the spaCy language model based on the provided language code. Falls back to a multilingual model if the
        specific model isn't available.

        Args:
            spacy_languages (dict[str, str]): Dictionary mapping language codes to spaCy model names.
            language (str): Language code.

        Returns:
            str | None: The name of the loaded spaCy model, or None if an error occurs.
        """
        try:
            return spacy_languages.get(language, "xx")
        except Exception as e:
            self.logger.error(
                f"Failed to load the language model for language '{language}': {e}"
            )
            return None
        
    def load_umap_model(
        self,
        n_neighbors: int = 15,
        n_components: int = 2,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
    ) -> UMAP | None:
        """
        Load the UMAP model with the specified parameters.

        Args:
            n_neighbors (int, optional): _Description_. Defaults to 15.
            n_components (int, optional): _description_. Defaults to 5.
            min_dist (float, optional): _description_. Defaults to 0.0.
            metric (str, optional): _description_. Defaults to "euclidean".
            random_state (int, optional): _description_. Defaults to 42.

        Returns:
            UMAP | None: The loaded UMAP model or None if an error occurs.
        """
        try:
            doc_count = len(self.docs)
            adjusted_components = min(n_components, max(1, doc_count - 1))
            if adjusted_components >= doc_count:
                self.logger.warning(
                    f"Too few documents ({doc_count}) for UMAP dimensionality reduction. Skipping topic modeling."
                )
                return None
            return UMAP(
                n_neighbors=n_neighbors,
                n_components=adjusted_components,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state,
            )
        except Exception as e:
            self.logger.error(f"Error loading UMAP model: {e}")
            return None

    def load_vectorizer_model(
        self, ngram_range: tuple = (1, 2), min_df: int = 2
    ) -> CountVectorizer | None:
        """
        Load the CountVectorizer model with the specified parameters.

        Args:
            ngram_range (tuple, optional): _description_. Defaults to (1, 2).
            min_df (int, optional): _description_. Defaults to 2.

        Returns:
            CountVectorizer | None: The loaded CountVectorizer model or None if an error occurs.
        """
        try:
            return CountVectorizer(
                stop_words=self.stop_words, ngram_range=ngram_range, min_df=min_df
            )
        except Exception as e:
            self.logger.error(f"Error loading vectorizer model: {e}")
            return None

    def load_representation_model(self) -> PartOfSpeech | None:
        """
        Load the representation model for BERTopic.

        Returns:
            PartOfSpeech | None: The loaded representation model or None if an error occurs.
        """
        try:
            return PartOfSpeech(self.nlp_name)
        except Exception as e:
            self.logger.error(f"Error loading representation model: {e}")
            return None

    def load_pipeline(
        self,
        top_n_words: int = 10,
        min_topic_size: int = 5,
        nr_topics: str = "auto",
        calculate_probabilities: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Load the complete pipeline for topic modeling.

        Args:
            top_n_words (int, optional): _description_. Defaults to 10.
            min_topic_size (int, optional): _description_. Defaults to 5.
            nr_topics (str, optional): _description_. Defaults to "auto".
            calculate_probabilities (bool, optional): _description_. Defaults to True.
            verbose (bool, optional): _description_. Defaults to True.
        """
        try:
            umap_model = self.load_umap_model()
            vectorizer_model = self.load_vectorizer_model()
            representation_model = self.load_representation_model()
            self.topic_model = BERTopic(
                language="multilingual",
                top_n_words=top_n_words,
                min_topic_size=min_topic_size,
                nr_topics=len(self.docs) // 15,
                calculate_probabilities=calculate_probabilities,
                umap_model=umap_model,
                vectorizer_model=vectorizer_model,
                representation_model=representation_model,
                verbose=verbose,
            )
        except Exception as e:
            logging.error(f"Error loading topic modeling pipeline: {e}")

    def fit_topic_model(self) -> pd.DataFrame | None:
        """
        Fit the topic model to the data and retrieve the topic information.

        Returns:
            pd.DataFrame | None: A DataFrame containing the topics' representations or None if an error occurs.
        """
        if len(self.docs) < 5:
            logging.warning("Not enough documents for topic modeling. Skipping fit.")
        try:
            if self.topic_model is not None:
                self.topic_model.fit_transform(documents=self.docs)
                self.topic_df = self.topic_model.get_topic_info()
                return self.topic_df
            else:
                logging.error("Topic model is not initialized.")
                return None
        except ValueError as e:
            logging.error(f"Error fitting topic model: {e}")
            return None

    def summarize_topics(
        self,
        language: str = "German",
    ) -> list[tuple[str, str]] | None:
        """
        Summarize the topics using zero-shot learning.
        NOTE: This method requires the ollama server to be running.

        Args:
            language (str): Language for the summary. Defaults to "German".

        Returns:
            list[tuple[str, str]] | None: A list of tuples containing topic titles and summaries, or None if an error occurs.
        """
        try:
            topic_titles: list[str] = []
            topic_summaries: list[str] = []

            if self.topic_df is None:
                self.get_topic_list()
            if self.topic_df is None:
                logging.error("Topic DataFrame is not available for summarization.")
                return None

            for keyword, doc in zip(
                self.topic_df["Representation"], self.topic_df["Representative_Docs"]
            ):
                # Generate title
                title_prompt = topic_titles_prompt.format(
                    language=language,
                    keywords=keyword,
                    docs=doc,
                )
                title = call_ollama(prompt=title_prompt)
                topic_titles.append(title if title is not None else "")

                # Generate summary using the generated title
                summary_prompt = topic_summaries_prompt.format(
                    language=language,
                    title=title,
                    keywords=keyword,
                    docs=doc,
                )
                summary = call_ollama(prompt=summary_prompt)
                topic_summaries.append(summary if summary is not None else "")

            return list(zip(topic_titles, topic_summaries))
        except Exception as e:
            logging.error(f"Error summarizing topics: {e}")
            return []
