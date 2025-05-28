import argparse
import logging
from pathlib import Path
from typing import Optional

from nextext import pipeline as ve
from nextext.modules import FileProcessor
from nextext.utils import setup_logging


setup_logging()


def parse_arguments(args_list: Optional[list] = None) -> argparse.Namespace:
    """
    Nextext pipelines ML models to transcribe, translate, and analyze natural language from audio/video.

    Args:
        args_list (list, optional): A list of arguments from the GUI to simulate command line input. Defaults to None.

    Returns:
        argparse.Namespace: Parsed arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser(
        description="Nextext turns voice into structured insight.",
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
        default="de",
        help="Specify the language code (ISO 639-1) of the source audio (default: 'de').",
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
    It also sets up logging and error handling for the entire pipeline.

    Raises:
        ValueError: If an invalid task is specified.
    """
    try:
        logging.info("\n\nInitiating Nextext...\n")

        # Parse command-line arguments
        args = parse_arguments()
        
        # Set up input/output directories and file paths
        file_processor = FileProcessor(args.file_path)

        # Transcribe and diarize the audio file
        if args.task in ["transcribe", "translate"]:
            transcript_df = ve.transcription_pipeline(
                file_path=args.file_path,
                src_lang=args.src_lang,
                model_id=args.model_id,
                task=args.task,
                api_key=ve.get_api_key() or "",
                speakers=args.speakers,
            )
        else:
            raise ValueError(
                "Invalid task. Please specify 'transcribe' or 'translate'."
            )

        # Machine translate the transcribed text
        if args.task == "translate" and args.trg_lang != "en":
            transcript_df = ve.translation_pipeline(
                df=transcript_df, trg_lang=args.trg_lang
            )

        # Streamline further data processing
        transcript_lang = args.src_lang if args.task == "transcribe" else args.trg_lang

        # Calculate word statistics
        if args.words:
            word_counts, named_entities, noun_sentiment, wordcloud_fig = (
                ve.wordlevel_pipeline(
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

        # Perform topic modeling
        if args.topics:
            topic_df = ve.topics_pipeline(
                data=transcript_df,
                language=transcript_lang,
            )
            file_processor.write_file_output(topic_df, "topics")

        # Summarize the transcribed text
        if args.summarize:
            transcript_summary = ve.summarization_pipeline(
                text=" ".join(transcript_df["text"].astype(str).tolist()),
                prompt_lang=transcript_lang,
            )
            file_processor.write_file_output(transcript_summary, "summary")

        # Classify text for toxicity
        if args.toxicity:
            transcript_df = ve.toxicity_pipeline(data=transcript_df)

        # Save final transcript
        file_processor.write_file_output(transcript_df, "transcript")

        logging.info(
            "The end of our elaborate plans, the end of everything that stands."
        )

    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
