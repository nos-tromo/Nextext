import numpy as np
import pandas as pd
import pycountry
import spacy
import torch
from bertopic import BERTopic
from bertopic.representation import PartOfSpeech
from camel_tools.tokenizers.word import simple_word_tokenize
from hdbscan import HDBSCAN
from loguru import logger
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from spacy.language import Language
from spacy.tokens import Doc
from umap import UMAP

from nextext.modules.ollama_cfg import OllamaPipeline
from nextext.utils.mappings_loader import load_mappings
from nextext.utils.spacy_model_loader import download_spacy_model


class TopicModeling:
    """
    TopicModeling provides an interface for topic modeling on text data using BERTopic.
    It supports language-specific preprocessing, embedding, dimensionality reduction, clustering,
    and topic representation, as well as zero-shot topic summarization via Ollama.

    Attributes:
        docs (list[str]): List of sentences to be processed.
        language (str): Language name for the text data.
        device (str): Device for computation ('cuda', 'mps', or 'cpu').
        nlp_name (str | None): spaCy model name for the specified language.
        stop_words (list[str]): List of stop words for the specified language.
        embedding_model (SentenceTransformer | None): Model for document embeddings.
        topic_model (BERTopic | None): BERTopic model instance.
        topic_df (pd.DataFrame | None): DataFrame containing topic information.

    Methods:
        _lang_code_to_name(lang_code: str) -> str | None:
            Convert a language code to its full language name.
        _load_spacy_model(spacy_languages: dict[str, str], language: str) -> str | None:
            Load the spaCy model name for the given language.
        _load_embedding_model(model: str = ...) -> SentenceTransformer | None:
            Load a SentenceTransformer embedding model.
        _load_umap_model(...) -> UMAP | None:
            Load a UMAP dimensionality reduction model.
        _load_hdbscan_model(...) -> HDBSCAN | None:
            Load an HDBSCAN clustering model.
        _load_vectorizer_model(ngram_range: tuple = (1, 2)) -> CountVectorizer | None:
            Load a CountVectorizer for tokenization.
        _load_representation_model() -> PartOfSpeech | None:
            Load a BERTopic representation model.
        load_pipeline(...) -> None:
            Initialize and configure the BERTopic pipeline.
        _embed_docs() -> np.ndarray | None:
            Embed documents using the embedding model.
        fit_topic_model() -> pd.DataFrame:
            Fit the topic model and return topic information.
        summarize_topics(language: str = "German") -> list[tuple[str, str]]:
            Summarize topics using zero-shot learning via Ollama.
    """

    def __init__(
        self,
        data: str | list[str] | pd.DataFrame,
        column: str | None = "text",
        lang_code: str = "en",
        spacy_language_file: str = "spacy_models.json",
    ) -> None:
        """
        Initialize the TopicModeling object.

        Args:
            data (str | list[str] | pd.DataFrame): List of text rows to be processed.
            column (str | None, optional): Column name in the DataFrame containing text data. Defaults to "text".
            lang_code (str, optional): Language code for the text data. Defaults to "en".
            spacy_language_file (str, optional): Path to the JSON file containing spaCy language model mappings. Defaults to "spacy_models.json".

        Raises:
            ValueError: If data is not a string, list of strings, or a DataFrame.
            ValueError: If the input data is empty.
        """
        # Validate input data type and convert to a list of sentences
        if isinstance(data, str):
            logger.info("Input data is a string. Tokenizing sentences.")
        elif isinstance(data, list):
            logger.info("Input data is a list of strings. Tokenizing sentences.")
            data = " ".join(data)
        elif isinstance(data, pd.DataFrame):
            logger.info(
                "Input data is a DataFrame. Extracting column '%s' for processing.",
                column,
            )
            data = " ".join(data[column].astype(str).tolist())
        else:
            logger.error(
                "Input data must be a string, list of strings, or a DataFrame. Received: %s",
                type(data),
            )
            raise ValueError("Input data must be a string or a list of strings.")

        self.language = self._lang_code_to_name(lang_code) or "english"
        self.docs = sent_tokenize(data, self.language)
        if not self.docs:
            logger.error("Input data is empty. Cannot proceed with topic modeling.")
            raise ValueError("Input data is empty. Cannot proceed with topic modeling.")

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        spacy_languages = load_mappings(spacy_language_file)
        self.nlp = self._load_spacy_model(lang_code, spacy_languages)
        logger.info("Loaded spaCy model for language '%s': %s", lang_code, self.nlp)

        self.stop_words = stopwords.words(self.language, "english")
        self.embedding_model: SentenceTransformer | None = None
        self.topic_model: BERTopic | None = None
        self.topic_df: pd.DataFrame | None = None

    def _lang_code_to_name(self, lang_code: str) -> str | None:
        """
        Converts the language code to its language name to be handled by the BERTopic object.

        Args:
            lang_code (str): Language code to be converted.

        Returns:
            str | None: The language name as a string, or None if the conversion fails.
        """
        try:
            lang_obj = pycountry.languages.get(alpha_2=lang_code.lower())
            if lang_obj and hasattr(lang_obj, "name"):
                return lang_obj.name.lower()
            else:
                logger.error("Could not find language name for code: %s", lang_code)
                return None
        except Exception as e:
            logger.error("Error converting language: %s.", e, exc_info=True)
            return None

    def _load_spacy_model(
        self, language: str, spacy_languages: dict[str, str]
    ) -> Language | None:
        """
        Load the spaCy model for the specified language code.

        Args:
            language (str): Language code for which to load the spaCy model.
            spacy_languages (dict[str, str]): Mapping of language codes to spaCy model names.

        Returns:
            Language | None: Loaded spaCy model or None if loading fails.
        """
        try:
            if language == "ar":
                nlp = spacy.blank("ar")
                nlp.tokenizer = lambda text, nlp=nlp: Doc(
                    nlp.vocab, words=simple_word_tokenize(text)
                )
                return nlp
            # Add other language-specific handling if needed
            model_id = None
            if language in spacy_languages.keys():
                model_id = spacy_languages.get(language)
            if model_id is None:
                logger.warning(
                    "Language '%s' not found in spaCy mappings. Using multilingual model.",
                    language,
                )
                model_id = spacy_languages.get("xx")
            if model_id is not None:
                download_spacy_model(model_id)
                return spacy.load(model_id)
            else:
                logger.error(
                    "No valid spaCy model id found for language '%s'.",
                    language,
                )
                return None
        except Exception as e:
            logger.warning(
                "Failed to load the language model for language '%s': %s", language, e
            )
            return None

    def _load_embedding_model(
        self, model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ) -> SentenceTransformer | None:
        """
        Load the SentenceTransformer model for encoding documents.

        Args:
            model (str, optional): The name of the SentenceTransformer model to load.
            Defaults to "sentence-transformers/paraphrase-multilingual-mpnet-base-v2".

        Returns:
            SentenceTransformer | None: The loaded SentenceTransformer model or None if an error occurs.
        """
        try:
            return SentenceTransformer(model, device=self.device, cache_folder="models")
        except Exception as e:
            logger.error("Error loading sentence transformer model: %s", e)
            return None

    def _load_umap_model(
        self,
        n_neighbors: int = 15,
        n_components: int = 5,
        min_dist: float = 0.0,
        metric: str = "cosine",
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
            adjusted_components = max(2, min(n_components, doc_count - 2))

            if adjusted_components >= doc_count:
                logger.warning(
                    "Too few documents (%d) for UMAP. Skipping topic modeling.",
                    doc_count,
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
            logger.error("Error loading UMAP model: %s", e)
            return None

    def _load_hdbscan_model(
        self,
        min_cluster_size=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    ) -> HDBSCAN | None:
        """
        Load the HDBSCAN model with the specified parameters.

        Args:
            min_cluster_size (int, optional): _description_. Defaults to 10.
            metric (str, optional): _description_. Defaults to "euclidean".
            cluster_selection_method (str, optional): _description_. Defaults to "eom".
            prediction_data (bool, optional): _description_. Defaults to True.

        Returns:
            HDBSCAN | None: The loaded HDBSCAN model or None if an error occurs.
        """
        try:
            return HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric=metric,
                cluster_selection_method=cluster_selection_method,
                prediction_data=prediction_data,
            )
        except Exception as e:
            logger.error("Error loading HDBSCAN model: %s", e)
            return None

    def _load_vectorizer_model(
        self, ngram_range: tuple = (1, 2)
    ) -> CountVectorizer | None:
        """
        Load the CountVectorizer model with the specified parameters.

        Args:
            ngram_range (tuple, optional): _description_. Defaults to (1, 2).

        Returns:
            CountVectorizer | None: The loaded CountVectorizer model or None if an error occurs.
        """
        try:
            return CountVectorizer(stop_words=self.stop_words, ngram_range=ngram_range)
        except Exception as e:
            logger.error("Error loading vectorizer model: %s", e)
            return None

    def _load_representation_model(self) -> PartOfSpeech | None:
        """
        Load the representation model for BERTopic.

        Returns:
            PartOfSpeech | None: The loaded representation model or None if an error occurs.
        """
        try:
            if self.nlp is not None:
                return PartOfSpeech(self.nlp)
            else:
                logger.error(
                    "spaCy model name is None. Cannot load PartOfSpeech representation model."
                )
                return None
        except Exception as e:
            logger.error("Error loading representation model: %s", e)
            return None

    def load_pipeline(
        self,
        min_topic_size: int = 15,
        calculate_probabilities: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Load the complete pipeline for topic modeling.

        Args:
            min_topic_size (int, optional): _description_. Defaults to 15.
            calculate_probabilities (bool, optional): _description_. Defaults to True.
            verbose (bool, optional): _description_. Defaults to True.
        """
        try:
            self.embedding_model = self._load_embedding_model()
            umap_model = self._load_umap_model()
            hdbscan_model = self._load_hdbscan_model()
            vectorizer_model = self._load_vectorizer_model()
            representation_model = self._load_representation_model()
            if (
                self.embedding_model is None
                or umap_model is None
                or hdbscan_model is None
                or vectorizer_model is None
                or representation_model is None
            ):
                logger.error(
                    "Failed to load one or more models. Cannot initialize BERTopic."
                )
                return
            if len(self.docs) < 5:
                logger.warning(
                    "Not enough documents for topic modeling. Skipping pipeline load."
                )
                return

            # Initialize the BERTopic model with the loaded components
            self.topic_model = BERTopic(
                language="multilingual",
                min_topic_size=min_topic_size,
                calculate_probabilities=calculate_probabilities,
                embedding_model=self.embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                representation_model=representation_model,
                verbose=verbose,
            )
        except Exception as e:
            logger.error("Error loading topic modeling pipeline: %s", e)

    def _embed_docs(self) -> np.ndarray | None:
        """
        Embed the documents using the SentenceTransformer model.

        Returns:
            list[list[float]]: A list of embeddings for each document.
        """
        if self.topic_model is None or self.embedding_model is None:
            logger.error("Topic model or embedding model is not initialized.")
            return None
        try:
            return self.embedding_model.encode(
                self.docs, show_progress_bar=True, device=self.device
            )
        except Exception as e:
            logger.error("Error embedding documents: %s", e)
            return None

    def fit_topic_model(self) -> pd.DataFrame:
        """
        Fit the topic model to the data and retrieve the topic information.

        Returns:
            pd.DataFrame: A DataFrame containing the topics' representations.

        Raises:
            ValueError: If the topic model is not initialized or if there are not enough documents.
        """
        if len(self.docs) < 5:
            logger.warning("Not enough documents for topic modeling. Skipping fit.")
        if self.topic_model is not None:
            # Ensure the embedding model is loaded
            if self.embedding_model is None:
                self.embedding_model = self._load_embedding_model()
            # Embed the documents
            embeddings = self._embed_docs()
            if embeddings is None:
                logger.error("Failed to embed documents. Cannot fit topic model.")
                return pd.DataFrame()
            # Fit the topic model
            self.topic_model.fit_transform(documents=self.docs, embeddings=embeddings)
            self.topic_df = self.topic_model.get_topic_info()
            return self.topic_df
        else:
            logger.error("Topic model is not initialized.")
            return pd.DataFrame()

    def summarize_topics(
        self,
        ollama_pipeline: OllamaPipeline,
    ) -> list[tuple[str, str]]:
        """
        Summarize the topics using zero-shot learning.
        NOTE: This method requires the ollama server to be running.

        Args:
            language (str): Language for the summary. Defaults to "German".

        Returns:
            list[tuple[str, str]]: A list of tuples containing topic titles and summaries.
        """
        topic_titles: list[str] = []
        topic_summaries: list[str] = []

        if self.topic_df is None or self.topic_df.empty:
            logger.error("Topic DataFrame is not available for summarization.")
            return []

        for keyword, doc in zip(
            self.topic_df["Representation"], self.topic_df["Representative_Docs"]
        ):
            # Generate title
            title_prompt = ollama_pipeline.load_prompt("topic_title").format(
                keywords=keyword,
                docs=doc,
            )
            title = ollama_pipeline.call_ollama_server(prompt=title_prompt)
            topic_titles.append(title if title is not None else "")

            # Generate summary using the generated title
            summary_prompt = ollama_pipeline.load_prompt("topic_summary").format(
                title=title if title is not None else "",
                keywords=keyword,
                docs=doc,
            )
            summary = ollama_pipeline.call_ollama_server(prompt=summary_prompt)
            topic_summaries.append(summary if summary is not None else "")

        return list(zip(topic_titles, topic_summaries))
