"""Streamlit UI for the Nextext audio analysis workflow."""

import io
import sys
import tempfile
import zipfile
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import altair as alt
import pandas as pd  # type: ignore[import-untyped]
import pycountry
import streamlit as st
from matplotlib.figure import Figure
from streamlit.web import cli as st_cli

from nextext.core.openai_cfg import InferencePipeline
from nextext.pipeline import (
    hate_speech_pipeline,
    normalize_language_code,
    summarization_pipeline,
    transcription_pipeline,
    translation_pipeline,
    wordlevel_pipeline,
)
from nextext.utils.env_cfg import set_offline_env
from nextext.utils.log_cfg import setup_logging
from nextext.utils.mappings_loader import kv_to_vk, load_and_sort_mappings
from nextext.utils.model_registry import flush_gpu

set_offline_env()
setup_logging()
PIPELINE_STAGE_LABELS: tuple[str, ...] = (
    "Transcribing",
    "Translating",
    "Running word-level analysis",
    "Summarizing",
    "Detecting hate speech",
)


def _dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to XLSX bytes using openpyxl.

    Args:
        df (pd.DataFrame): The DataFrame to serialize.

    Returns:
        bytes: XLSX bytes suitable for inclusion in a ZIP archive.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(cast(Any, buffer), engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return buffer.getvalue()


def _build_results_archive(
    results: Sequence[dict[str, Any]],
    archive_stem: str,
) -> bytes:
    """Bundle every produced output for every processed file into a ZIP.

    Outputs mirror the CLI labels (``transcript``, ``summary``, ``words``,
    ``entities``, ``wordcloud``, ``hate_speech``). All files are wrapped in
    a top-level directory matching the ZIP filename stem, then nested inside
    a per-upload subdirectory so single-file and multi-file archives share
    the same ``{archive_stem}/{original_stem}/{original_stem}_{output}.ext``
    layout.

    Args:
        results (Sequence[dict[str, Any]]): Stored result entries.
        archive_stem (str): Top-level directory name inside the ZIP, matching
            the download filename stem.

    Returns:
        bytes: ZIP archive bytes ready to feed into ``st.download_button``.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for index, result in enumerate(results, start=1):
            original_name = str(result.get("file_name", f"result_{index}"))
            stem = Path(original_name).stem or f"result_{index}"
            base = f"{archive_stem}/{stem}/{stem}"

            transcript = result.get("transcript")
            if isinstance(transcript, pd.DataFrame) and not transcript.empty:
                zf.writestr(
                    f"{base}_transcript.csv",
                    transcript.to_csv(index=False).encode("utf-8"),
                )
                zf.writestr(
                    f"{base}_transcript.xlsx",
                    _dataframe_to_excel_bytes(transcript),
                )

            summary = result.get("summary")
            if isinstance(summary, str) and summary.strip():
                zf.writestr(f"{base}_summary.txt", summary.encode("utf-8"))

            word_counts = result.get("word_counts")
            if isinstance(word_counts, pd.DataFrame) and not word_counts.empty:
                zf.writestr(
                    f"{base}_words.csv",
                    word_counts.to_csv(index=False).encode("utf-8"),
                )
                zf.writestr(
                    f"{base}_words.xlsx",
                    _dataframe_to_excel_bytes(word_counts),
                )

            named_entities = result.get("named_entities")
            if isinstance(named_entities, pd.DataFrame) and not named_entities.empty:
                zf.writestr(
                    f"{base}_entities.csv",
                    named_entities.to_csv(index=False).encode("utf-8"),
                )
                zf.writestr(
                    f"{base}_entities.xlsx",
                    _dataframe_to_excel_bytes(named_entities),
                )

            wordcloud = result.get("wordcloud")
            if isinstance(wordcloud, Figure):
                png_buffer = io.BytesIO()
                wordcloud.savefig(png_buffer, format="png", bbox_inches="tight")
                zf.writestr(f"{base}_wordcloud.png", png_buffer.getvalue())

            findings = result.get("hate_speech_findings")
            if findings:
                findings_df = pd.DataFrame(findings)
                zf.writestr(
                    f"{base}_hate_speech.csv",
                    findings_df.to_csv(index=False).encode("utf-8"),
                )
                zf.writestr(
                    f"{base}_hate_speech.xlsx",
                    _dataframe_to_excel_bytes(findings_df),
                )

    return buffer.getvalue()


def _results_archive_bytes(
    results: Sequence[dict[str, Any]],
    archive_stem: str,
) -> bytes:
    """Return cached archive bytes for the current results list.

    The bytes are memoized in ``st.session_state`` keyed by the identity of
    the results list and the archive stem, so switching tabs or selectors
    does not rebuild the archive on every rerun.

    Args:
        results (Sequence[dict[str, Any]]): Stored result entries.
        archive_stem (str): Top-level directory name inside the ZIP.

    Returns:
        bytes: ZIP archive bytes for the current results list.
    """
    cache_key = (id(results), archive_stem)
    cached = st.session_state.get("_results_archive")
    if isinstance(cached, tuple) and cached[0] == cache_key:
        return cached[1]
    payload = _build_results_archive(results, archive_stem)
    st.session_state["_results_archive"] = (cache_key, payload)
    return payload


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
        "named_entities": None,
        "wordcloud": None,
        "hate_speech_findings": None,
        "resolved_src_lang": file_opts["src_lang"],
        "transcript_language": transcript_language,
    }

    # Word-level analysis
    _notify(PIPELINE_STAGE_LABELS[2], 2)
    if file_opts["words"]:
        wc, ner, cloud = wordlevel_pipeline(
            df,
            normalize_language_code(transcript_language) or "en",
        )
        result["word_counts"] = wc
        result["named_entities"] = ner
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

    # Hate speech detection
    _notify(PIPELINE_STAGE_LABELS[4], 4)
    if file_opts.get("hate_speech"):
        if inference_pipeline is None:
            inference_pipeline = InferencePipeline(
                out_language=_language_name(transcript_language)
            )
            if not inference_pipeline.get_health():
                raise ConnectionError(
                    "The configured inference provider is not reachable. "
                    "Please ensure it is running and accessible."
                )
        result["hate_speech_findings"] = hate_speech_pipeline(
            df=df,
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
            flush_gpu()

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
    trg_lang_maps, trg_lang_names = load_and_sort_mappings(
        "translategemma_languages.json"
    )
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
        words = st.checkbox("Word-level analysis")
        summarization = st.checkbox("Summarisation")
        hate_speech = st.checkbox("Hate speech detection")
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
        task=task,
        speakers=speakers,
        words=words,
        summarization=summarization,
        hate_speech=hate_speech,
    )

    if run and uploaded_files:
        results = _process_uploaded_files(uploaded_files, st.session_state["opts"])
        st.session_state.pop("_results_archive", None)
        st.session_state["results"] = results
        st.session_state["result"] = results[0]
        st.session_state["selected_result_file"] = results[0]["file_name"]
        st.session_state["results_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.success("Done! Select another tab to view results.")


def main() -> None:
    """Main function to run the Streamlit app (tab‑based navigation)."""
    st.set_page_config(page_title="Nextext", layout="wide")
    st.title("Nextext")
    st.subheader("Transcribe, translate, and analyze audio/video files")

    # Top‑level tabs
    tab_params, tab_transcript, tab_summary, tab_words, tab_hate = st.tabs(
        ["Parameters", "Transcript", "Summary", "Word-level Analysis", "Hate Speech"]
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
        for t in (tab_transcript, tab_summary, tab_words, tab_hate):
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

    archive_timestamp = st.session_state.get(
        "results_timestamp"
    ) or datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_stem = f"{archive_timestamp}_nextext_output"
    st.download_button(
        label="⬇️ Download all outputs (ZIP)",
        data=_results_archive_bytes(results, archive_stem),
        file_name=f"{archive_stem}.zip",
        mime="application/zip",
        help=(
            "Bundles every produced output (transcripts, summaries, word "
            "tables, named entities, word clouds, hate-speech findings) for "
            "every processed file."
        ),
    )

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
            _ner_df = result["named_entities"]
            if _ner_df is not None and not _ner_df.empty:
                _top15 = (
                    _ner_df.sort_values("Frequency", ascending=False).head(15).copy()
                )
                _top15["Label"] = _top15["Entity"] + " (" + _top15["Category"] + ")"
                st.altair_chart(
                    alt.Chart(_top15)
                    .mark_bar()
                    .encode(
                        x=alt.X("Label:N", sort="-y", title="Entity"),
                        y=alt.Y("Frequency:Q", title="Frequency"),
                        tooltip=["Category", "Entity", "Frequency"],
                    )
                    .properties(height=350),
                    width="stretch",
                )
                _categories = sorted(_ner_df["Category"].unique().tolist())
                _ner_col1, _ner_col2 = st.columns([1, 2])
                with _ner_col1:
                    _selected_cat = st.selectbox(
                        "Entity category",
                        options=["All", *_categories],
                        index=0,
                        key="ner_category_select",
                    )
                _filtered = (
                    _ner_df
                    if _selected_cat == "All"
                    else _ner_df[_ner_df["Category"] == _selected_cat]
                )
                _filtered = _filtered.sort_values("Frequency", ascending=False)
                _entity_labels = {
                    row["Entity"]: f"{row['Entity']} ({row['Frequency']} mentions)"
                    for _, row in _filtered.iterrows()
                }
                with _ner_col2:
                    _selected_entity = st.selectbox(
                        "Entity",
                        options=_filtered["Entity"].tolist(),
                        format_func=lambda e: _entity_labels.get(e, e),
                        key="ner_entity_select",
                    )
                if _selected_entity:
                    _transcript_df = result["transcript"]
                    _mask = _transcript_df["text"].str.contains(
                        _selected_entity, case=False, na=False
                    )
                    _passages = _transcript_df[_mask]
                    if not _passages.empty:
                        _display_cols = [
                            c
                            for c in ("start", "end", "speaker", "text")
                            if c in _passages.columns
                        ]
                        st.dataframe(
                            _passages[_display_cols].reset_index(drop=True),
                            hide_index=True,
                        )
                    else:
                        st.info(
                            f"No transcript passages found for '{_selected_entity}'."
                        )
            else:
                st.info("No named entities found.")

            st.subheader("☁️ Word Cloud")
            st.pyplot(result["wordcloud"])

    # ------------ Hate Speech tab -------------------
    with tab_hate:
        findings = result.get("hate_speech_findings")
        if findings is None:
            st.warning("Hate speech detection not requested.")
        elif not findings:
            st.success("No hate speech detected.")
        else:
            st.subheader("🚨 Hate Speech Findings")
            for item in findings:
                with st.expander(
                    f"{item['category'].title()} — {item['confidence']} confidence"
                ):
                    st.write(f"**Reason:** {item['reason']}")
                    st.write(f"**Flagged text:** {item['text']}")

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
