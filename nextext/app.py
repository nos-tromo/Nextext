"""Streamlit UI for the Nextext audio analysis workflow."""

import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import altair as alt
import pycountry
import streamlit as st
import streamlit.components.v1 as components
from streamlit.web import cli as st_cli

from nextext.modules.openai_cfg import InferencePipeline
from nextext.pipeline import (
    normalize_language_code,
    summarization_pipeline,
    transcription_pipeline,
    translation_pipeline,
    wordlevel_pipeline,
)
from nextext.utils.env_cfg import set_offline_env
from nextext.utils.log_cfg import setup_logging
from nextext.utils.mappings_loader import kv_to_vk, load_and_sort_mappings

set_offline_env()
setup_logging()
PIPELINE_STAGE_LABELS: tuple[str, ...] = (
    "Transcribing",
    "Translating",
    "Running word-level analysis",
    "Summarizing",
)


def _language_name(lang_code: str | None) -> str:
    """Convert an ISO language code to a human-readable name for LLM output settings.

    Args:
        lang_code (str | None): The ISO 639-1 language code.

    Returns:
        str: The human-readable language name, or "German" if the code is None.
    """
    if not lang_code:
        return "German"
    lang = pycountry.languages.get(alpha_2=normalize_language_code(lang_code))
    return lang.name if lang is not None else lang_code


def _default_target_language(
    language_maps: dict[str, str],
    language_names: Sequence[str],
) -> tuple[int, str]:
    """Resolve a safe default target language for the translation UI.

    Args:
        language_maps (dict[str, str]): Mapping of language codes to labels.
        language_names (Sequence[str]): Sorted language labels shown in the UI.

    Returns:
        tuple[int, str]: The default selectbox index and language code.
    """
    preferred_codes = ("de-DE", "de", "en")
    for language_code in preferred_codes:
        language_name = language_maps.get(language_code)
        if language_name in language_names:
            return language_names.index(language_name), language_code

    if not language_names:
        return 0, "en"

    fallback_name = language_names[0]
    fallback_code = kv_to_vk(language_maps).get(fallback_name, "en")
    return 0, fallback_code


def _progress_value(
    file_index: int,
    total_files: int,
    stage_index: int,
    total_stages: int,
) -> float:
    """Calculate overall progress for sequential multi-file processing.

    Args:
        file_index (int): One-based file index currently being processed.
        total_files (int): Total number of files in the run.
        stage_index (int): Zero-based pipeline stage index.
        total_stages (int): Total number of pipeline stages.

    Returns:
        float: A normalized progress value between 0.0 and 1.0.

    Raises:
        ValueError: If any numeric input is outside the supported range.
    """
    if total_files <= 0:
        raise ValueError("total_files must be greater than zero.")
    if total_stages <= 0:
        raise ValueError("total_stages must be greater than zero.")
    if file_index < 1 or file_index > total_files:
        raise ValueError("file_index must be within the file count.")
    if stage_index < 0 or stage_index >= total_stages:
        raise ValueError("stage_index must be within the stage count.")

    completed_files = file_index - 1
    stage_progress = stage_index / total_stages
    return (completed_files + stage_progress) / total_files


def _result_file_names(results: Sequence[dict[str, Any]]) -> list[str]:
    """Return stable labels for processed result entries.

    Args:
        results (Sequence[dict[str, Any]]): Stored result entries.

    Returns:
        list[str]: File names displayed in the result selector.
    """
    file_names: list[str] = []
    for index, result in enumerate(results, start=1):
        file_names.append(str(result.get("file_name", f"File {index}")))
    return file_names


def _download_file_name(result: dict[str, Any], suffix: str) -> str:
    """Build a download filename based on the original uploaded file name.

    Args:
        result (dict[str, Any]): Stored result payload for one file.
        suffix (str): Output suffix including the desired extension.

    Returns:
        str: A download filename derived from the original upload name.
    """
    original_name = str(result.get("file_name", "result"))
    original_stem = Path(original_name).stem or "result"
    return f"{original_stem}_{suffix}"


def _select_result(
    results: Sequence[dict[str, Any]],
    selected_file_name: str | None,
) -> dict[str, Any]:
    """Select the requested result entry or fall back to the first one.

    Args:
        results (Sequence[dict[str, Any]]): Stored result entries.
        selected_file_name (str | None): Requested file name.

    Returns:
        dict[str, Any]: The matching result entry.

    Raises:
        ValueError: If no results are available.
    """
    if not results:
        raise ValueError("At least one result entry is required.")

    if selected_file_name:
        for result in results:
            if result.get("file_name") == selected_file_name:
                return result
    return dict(results[0])


def _run_pipeline(
    tmp_file: Path,
    opts: dict[str, Any],
    status_callback: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    """Run the core pipeline for one file and return its result payload.

    Args:
        tmp_file (Path): Path to the temporary file.
        opts (dict[str, Any]): Options for the pipeline.
        status_callback (Callable[[str, int], None] | None): Optional
            callback invoked before each pipeline stage.

    Returns:
        dict[str, Any]: A result payload for one processed file.

    Raises:
        ConnectionError: If the configured inference provider is not
            reachable.
    """
    file_opts = dict(opts)

    def _notify(stage_name: str, stage_index: int) -> None:
        """Emit progress updates when a callback is configured.

        Args:
            stage_name (str): Human-readable stage label.
            stage_index (int): Zero-based pipeline stage index.
        """
        if status_callback is not None:
            status_callback(stage_name, stage_index)

    # Transcription
    _notify(PIPELINE_STAGE_LABELS[0], 0)
    df, updated_src_lang = transcription_pipeline(
        file_path=tmp_file,
        trg_lang=file_opts["trg_lang"],
        src_lang=file_opts["src_lang"],
        model_id=file_opts["model_id"],
        task=file_opts["task"],
        n_speakers=file_opts["speakers"],
    )
    file_opts["src_lang"] = updated_src_lang

    # Translation
    _notify(PIPELINE_STAGE_LABELS[1], 1)
    if file_opts["task"] == "translate" and file_opts["trg_lang"] != "en":
        inference_pipeline = InferencePipeline(
            out_language=_language_name(file_opts["trg_lang"])
        )
        if not inference_pipeline.get_health():
            raise ConnectionError(
                "The configured inference provider is not reachable. "
                "Please ensure it is running and accessible."
            )
        df = translation_pipeline(
            df,
            file_opts["trg_lang"],
            src_lang=file_opts["src_lang"],
            inference_pipeline=inference_pipeline,
        )
    else:
        inference_pipeline = None

    # Store the DataFrame and default values in the result payload
    transcript_language = file_opts[
        "trg_lang" if file_opts["task"] == "translate" else "src_lang"
    ]
    result: dict[str, Any] = {
        "transcript": df,
        "summary": None,
        "word_counts": None,
        "noun_sentiment": None,
        "noun_graph": None,
        "named_entities": None,
        "wordcloud": None,
        "resolved_src_lang": file_opts["src_lang"],
        "transcript_language": transcript_language,
    }

    # Word-level analysis
    _notify(PIPELINE_STAGE_LABELS[2], 2)
    if file_opts["words"]:
        wc, ner, nouns, graph, cloud = wordlevel_pipeline(
            df,
            normalize_language_code(transcript_language) or "en",
        )
        result["word_counts"] = wc
        result["named_entities"] = ner
        result["noun_sentiment"] = nouns
        result["noun_graph"] = graph
        result["wordcloud"] = cloud

    # Summarization
    _notify(PIPELINE_STAGE_LABELS[3], 3)
    if file_opts["summarization"]:
        if inference_pipeline is None:
            inference_pipeline = InferencePipeline(
                out_language=_language_name(transcript_language)
            )
            if not inference_pipeline.get_health():
                raise ConnectionError(
                    "The configured inference provider is not reachable. "
                    "Please ensure it is running and accessible."
                )
        result["summary"] = summarization_pipeline(
            " ".join(df["text"].astype(str).tolist()),
            inference_pipeline=inference_pipeline,
        )

    return result


def _process_uploaded_files(
    uploaded_files: Sequence[Any],
    opts: dict[str, Any],
) -> list[dict[str, Any]]:
    """Process uploaded files sequentially with shared settings.

    Args:
        uploaded_files (Sequence[Any]): Uploaded file objects from Streamlit.
        opts (dict[str, Any]): Shared pipeline settings for the run.

    Returns:
        list[dict[str, Any]]: Result payloads in upload order.
    """
    total_files = len(uploaded_files)
    total_stages = len(PIPELINE_STAGE_LABELS)
    progress_bar = st.progress(0.0, text="Preparing files…")
    results: list[dict[str, Any]] = []

    for file_index, uploaded_file in enumerate(uploaded_files, start=1):
        file_name = str(getattr(uploaded_file, "name", f"File {file_index}"))

        def _update_progress(stage_name: str, stage_index: int) -> None:
            """Update the multi-file progress bar.

            Args:
                stage_name (str): Human-readable stage label.
                stage_index (int): Zero-based pipeline stage index.
            """
            progress_bar.progress(
                _progress_value(
                    file_index=file_index,
                    total_files=total_files,
                    stage_index=stage_index,
                    total_stages=total_stages,
                ),
                text=(
                    f"Processing file {file_index}/{total_files}: "
                    f"{file_name} ({stage_name})"
                ),
            )

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(file_name).suffix,
            ) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = Path(tmp.name)
            file_result = _run_pipeline(
                tmp_path,
                opts,
                status_callback=_update_progress,
            )
            file_result["file_name"] = file_name
            results.append(file_result)
            progress_bar.progress(
                file_index / total_files,
                text=f"Completed file {file_index}/{total_files}: {file_name}",
            )
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()

    progress_bar.progress(
        1.0,
        text=f"Processed {total_files} file(s).",
    )
    return results


def _start_page() -> None:
    """Render the pipeline settings form in the Parameters tab."""
    st.markdown("## ⚙️ Pipeline settings")

    uploaded_files = st.file_uploader(
        "Audio / video file(s)",
        type=["mp3", "m4a", "mp4", "mkv", "ogg", "wav", "webm"],
        accept_multiple_files=True,
    )

    # Load source language mappings from Whisper and target language mappings for translation.
    src_lang_maps, src_lang_names = load_and_sort_mappings("whisper_languages.json")
    trg_lang_maps, trg_lang_names = load_and_sort_mappings("translation_languages.json")
    default_trg_lang_index, default_trg_lang_code = _default_target_language(
        language_maps=trg_lang_maps,
        language_names=trg_lang_names,
    )

    # Create GUI columns and widgets
    task = st.radio("Task", ["transcribe", "translate"], horizontal=True)
    col1, col2 = st.columns(2)
    with col1:
        src_lang_name = st.selectbox(
            "Source language",
            ["Detect language"] + src_lang_names,
            index=0,
        )
        model_id = st.selectbox(
            "Whisper model",
            ["default", "large-v3", "large-v2", "medium", "small", "base", "tiny"],
            index=0,
        )
        words = st.checkbox("Word-level analysis")
        summarization = st.checkbox("Summarisation")
    with col2:
        trg_lang_name = st.selectbox(
            "Target language (for translate task)",
            trg_lang_names,
            index=default_trg_lang_index,
        )
        speakers = st.number_input("Max speakers", 1, 10, value=1, step=1)

    src_lang_code = kv_to_vk(src_lang_maps).get(src_lang_name)
    trg_lang_code = kv_to_vk(trg_lang_maps).get(
        trg_lang_name,
        default_trg_lang_code,
    )

    run = st.button("▶️ Run", disabled=not uploaded_files)

    # Persist options for the run
    st.session_state["opts"] = dict(
        src_lang=src_lang_code,
        trg_lang=trg_lang_code,
        model_id=model_id,
        task=task,
        speakers=speakers,
        words=words,
        summarization=summarization,
    )

    if run and uploaded_files:
        results = _process_uploaded_files(uploaded_files, st.session_state["opts"])
        st.session_state["results"] = results
        st.session_state["result"] = results[0]
        st.session_state["selected_result_file"] = results[0]["file_name"]
        st.success("Done! Select another tab to view results.")


def main() -> None:
    """Main function to run the Streamlit app (tab‑based navigation)."""
    st.set_page_config(page_title="Nextext", layout="wide")
    st.title("Nextext")
    st.subheader("Transcribe, translate, and analyze audio/video files")

    # Top‑level tabs
    tab_params, tab_transcript, tab_summary, tab_words = st.tabs(
        ["Parameters", "Transcript", "Summary", "Word-level Analysis"]
    )

    # ---------------- Parameters tab ----------------
    with tab_params:
        _start_page()

    # Helper: guard for tabs that need results
    results = st.session_state.get("results")
    if results is None and "result" in st.session_state:
        results = [st.session_state["result"]]

    if not results:
        msg = (
            "After you upload one or more files and press **Run** in the "
            "Parameters tab, the results will appear here."
        )
        for t in (tab_transcript, tab_summary, tab_words):
            with t:
                st.info(msg)

        # Footer
        st.markdown(
            """
            <hr style="margin-top:2rem;margin-bottom:1rem;">
            <p style="text-align:center;">
                🔗 <a href="https://github.com/nos-tromo/Nextext" target="_blank">GitHub</a>
            </p>
            """,
            unsafe_allow_html=True,
        )
        return

    selected_file_name = None
    if len(results) > 1:
        file_names = _result_file_names(results)
        current_file_name = st.session_state.get(
            "selected_result_file",
            file_names[0],
        )
        if current_file_name not in file_names:
            current_file_name = file_names[0]
        selected_file_name = st.selectbox(
            "Processed file",
            file_names,
            index=file_names.index(current_file_name),
        )
        st.session_state["selected_result_file"] = selected_file_name

    result = _select_result(results, selected_file_name)

    # ---------------- Transcript tab ----------------
    with tab_transcript:
        st.subheader("📜 Transcript")
        if "file_name" in result:
            st.caption(f"Showing results for `{result['file_name']}`")
        st.dataframe(result["transcript"], hide_index=True)

    # ---------------- Summary tab -------------------
    with tab_summary:
        if result["summary"] is None:
            st.warning("Summary not requested.")
        else:
            st.subheader("📝 Summary")
            st.write(result["summary"])
            st.download_button(
                label="Download",
                data=result["summary"],
                file_name=_download_file_name(result, "summary.txt"),
                mime="text/plain",
            )

    # ------------ Word‑level analysis tab -----------
    with tab_words:
        if result["word_counts"] is None:
            st.warning("Word statistics not requested.")
        else:
            st.subheader("🔠 Top Words")
            wc_df = result["word_counts"]
            word_col = wc_df.columns[0]
            freq_col = wc_df.columns[1]
            chart = (
                alt.Chart(wc_df.reset_index())
                .mark_bar()
                .encode(
                    x=alt.X(f"{word_col}:N", sort="-y", title=word_col),
                    y=alt.Y(f"{freq_col}:Q", title=freq_col),
                    tooltip=[word_col, freq_col],
                )
                .properties(height=400)
            )
            st.altair_chart(chart, width="stretch")

            st.subheader("🧩 Named Entities")
            st.dataframe(result["named_entities"], hide_index=True)

            st.subheader("🗣️ Noun Sentiment")
            st.dataframe(result["noun_sentiment"], hide_index=True)
            if result["noun_graph"]:
                components.html(result["noun_graph"], height=800, scrolling=False)
            else:
                st.info("No noun graph available.")

            st.subheader("☁️ Word Cloud")
            st.pyplot(result["wordcloud"])

    # ---------------- Footer -----------------------
    st.markdown(
        """
        <hr style="margin-top:2rem;margin-bottom:1rem;">
        <p style="text-align:center;">
            🔗 <a href="https://github.com/nos-tromo/nextext" target="_blank">GitHub&nbsp;Repository</a>
        </p>
        """,
        unsafe_allow_html=True,
    )


# ---- Streamlit CLI wrapper ----------------------------------------------- #
def cli() -> None:
    """CLI entry point for the Streamlit app. This function is used to run the app from the command
    line. It sets up the command line arguments as if the user typed them. For example: `streamlit
    run app.py <any extra args>`.
    """
    sys.argv = ["streamlit", "run", __file__] + sys.argv[1:]
    sys.exit(st_cli.main())


if __name__ == "__main__":
    main()
