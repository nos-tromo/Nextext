import argparse
from pathlib import Path

from loguru import logger

from nextext.modules.ollama_cfg import OllamaPipeline
from nextext.modules.processing import FileProcessor
from nextext.pipeline import (
    get_api_key,
    hatespeech_pipeline,
    summarization_pipeline,
    topics_pipeline,
    transcription_pipeline,
    translation_pipeline,
    wordlevel_pipeline,
)
from nextext.utils.logging_cfg import init_logger

init_logger()


def parse_arguments(args_list: list | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the Nextext CLI.
    This function sets up the argument parser with various options for file processing,
    language settings, model selection, and analysis features.
    It returns the parsed arguments as an `argparse.Namespace` object.
    This function allows users to specify the audio file to be processed, the source and target languages,
    the model size for Whisper, the task to perform (transcription or translation),
    and various analysis options such as word statistics, topic modeling, summarization, and toxicity analysis.
    It also supports a full analysis mode that combines multiple analysis features into one command.
    It is designed to be flexible and user-friendly, providing default values for most options
    while allowing users to customize their processing pipeline as needed.

    Args:
        args_list (list | None, optional): Drop-in list of arguments to parse.
        If None, uses `sys.argv` to parse command-line arguments. Defaults to None.

    Raises:
        argparse.ArgumentError: If there is an error in argument parsing.

    Returns:
        argparse.Namespace: Parsed command-line arguments as a namespace object.
    """
    parser = argparse.ArgumentParser(
        description="Audio transcription and analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-f",
        "--file",
        dest="file_path",
        type=Path,
        required=True,
        help="Specify the file path and name of the audio file to be transcribed.",
    )
    parser.add_argument(
        "-sl",
        "--src-lang",
        dest="src_lang",
        type=str,
        default=None,
        help="Specify the language code (ISO 639-1) of the source audio (default: None).",
    )
    parser.add_argument(
        "-tl",
        "--trg-lang",
        dest="trg_lang",
        type=str,
        default="de",
        help="Specify the language code (ISO 639-1) of the target language (default: 'de').",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model_id",
        type=str,
        default="default",
        help="Specify the model size for Whisper (default: 'default' = 'turbo').",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        type=str,
        default="transcribe",
        help="Specify the task to perform: 'transcribe' (default), or 'translate'.",
    )
    parser.add_argument(
        "-s",
        "--speakers",
        dest="speakers",
        type=int,
        default=1,
        help="Specify the maximum number of speakers for diarization (default: 1).",
    )
    parser.add_argument(
        "-w",
        "--words",
        dest="words",
        action="store_true",
        help="Show most frequently used words (default: False).",
    )
    parser.add_argument(
        "-tm",
        "--topics",
        dest="topics",
        action="store_true",
        help="Enable topic modeling analysis (default: False).",
    )
    parser.add_argument(
        "-sum",
        "--summarize",
        dest="summarize",
        action="store_true",
        help="Additional text and topic summarization (default: False).",
    )
    parser.add_argument(
        "-tox",
        "--toxicity",
        dest="toxicity",
        action="store_true",
        help="Enable toxicity analysis (default: False).",
    )
    parser.add_argument(
        "-F",
        "--full-analysis",
        dest="full_analysis",
        action="store_true",
        help="Enable full analysis, equivalent to using -w -tm -sum -tox (default: False).",
    )

    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    if args.full_analysis:
        args.words = True
        args.topics = True
        args.summarize = True
        args.toxicity = True

    return args


def main() -> None:
    """
    Run the Nextext pipeline for transcription, translation, and analysis.
    This function orchestrates the entire process, including transcription, translation,
    word statistics, and topic modeling.
    It handles the command-line arguments and manages the flow of data through the various modules.

    Raises:
        ValueError: If an invalid task is specified.
        ConnectionError: If the Ollama server is not reachable for analysis tasks.
    """
    logger.info("\n\nInitiating Nextext...\n")

    # Parse command-line arguments
    args = parse_arguments()

    # Set up input/output directories and file paths
    file_processor = FileProcessor(args.file_path)

    # Transcribe and diarize the audio file
    if args.task in ["transcribe", "translate"]:
        transcript_df, updated_src_lang = transcription_pipeline(
            file_path=args.file_path,
            api_key=get_api_key() or "",
            trg_lang=args.trg_lang,
            src_lang=args.src_lang,
            model_id=args.model_id,
            task=args.task,
            n_speakers=args.speakers,
        )
        args.src_lang = updated_src_lang  # Update source language if detected
    else:
        logger.error("Invalid task specified: {}", args.task)
        raise ValueError("Invalid task. Please specify 'transcribe' or 'translate'.")

    # Machine translate the transcribed text
    if args.task == "translate" and args.trg_lang != "en":
        transcript_df = translation_pipeline(df=transcript_df, trg_lang=args.trg_lang)

    # Streamline language for further processing
    transcript_lang = args.src_lang if args.task == "transcribe" else args.trg_lang

    # Calculate word statistics
    if args.words:
        word_counts, named_entities, noun_sentiment, noun_graph, wordcloud_fig = (
            wordlevel_pipeline(
                data=transcript_df,
                language=transcript_lang,
            )
        )
        exports = [
            (word_counts, "words"),
            (named_entities, "entities"),
            (noun_sentiment, "nouns"),
            (wordcloud_fig, "wordcloud"),
        ]
        for export, name in exports:
            file_processor.write_file_output(export, name)

    if args.topics or args.summarize or args.toxicity:
        ollama_pipeline = OllamaPipeline()
        if not ollama_pipeline._get_ollama_health():
            logger.error("Ollama server is not reachable.")
            raise ConnectionError(
                "Ollama server is not reachable. Please ensure it is running and accessible."
            )

        # Perform topic modeling
        if args.topics:
            topic_df = topics_pipeline(
                data=transcript_df,
                language=transcript_lang,
                ollama_pipeline=ollama_pipeline,
            )
            if topic_df is not None:
                file_processor.write_file_output(topic_df, "topics")

        # Summarize the transcribed text
        if args.summarize:
            transcript_summary = summarization_pipeline(
                text=" ".join(transcript_df["text"].astype(str).tolist()),
                ollama_pipeline=ollama_pipeline,
            )
            if transcript_summary is not None:
                file_processor.write_file_output(transcript_summary, "summary")

        # Classify text for hate speech
        if args.toxicity:
            transcript_df = hatespeech_pipeline(
                ollama_pipeline=ollama_pipeline, df=transcript_df
            )

    # Save final transcript
    file_processor.write_file_output(transcript_df, "transcript")

    logger.info("The end of our elaborate plans, the end of everything that stands.")


if __name__ == "__main__":
    main()
