"""Streamlit UI for the Nextext audio analysis workflow."""

import sys
import tempfile
from collections.abc import Sequence
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
from nextext.utils.logging_cfg import init_logger
from nextext.utils.mappings_loader import kv_to_vk, load_and_sort_mappings

init_logger()


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


def _run_pipeline(tmp_file: Path, opts: dict) -> None:
    """Run the core pipeline and stash results in st.session_state.

    Args:
        tmp_file (Path): Path to the temporary file.
        opts (dict): Options for the pipeline.

    Raises:
        ConnectionError: If the Ollama server is not reachable.
    """
    # Transcription
    with st.spinner("Transcribing… this might take a while ⏳"):
        df, updated_src_lang = transcription_pipeline(
            file_path=tmp_file,
            trg_lang=opts["trg_lang"],
            src_lang=opts["src_lang"],
            model_id=opts["model_id"],
            task=opts["task"],
            n_speakers=opts["speakers"],
        )
        # Update the source language in the session state
        opts["src_lang"] = updated_src_lang
        st.session_state["opts"] = opts

    # Translation
    with st.spinner("Translating… this might take a while ⏳"):
        if opts["task"] == "translate" and opts["trg_lang"] != "en":
            inference_pipeline = InferencePipeline(
                out_language=_language_name(opts["trg_lang"])
            )
            if not inference_pipeline.get_health():
                raise ConnectionError(
                    "The configured inference provider is not reachable. Please ensure it is running and accessible."
                )
            df = translation_pipeline(
                df,
                opts["trg_lang"],
                src_lang=opts["src_lang"],
                inference_pipeline=inference_pipeline,
            )
        else:
            inference_pipeline = None

    # Store the DataFrame and default values in session state
    result: dict[str, Any] = {
        "transcript": df,
        "summary": None,
        "word_counts": None,
        "noun_sentiment": None,
        "noun_graph": None,
        "named_entities": None,
        "wordcloud": None,
    }

    # Word-level analysis
    with st.spinner("Running word-level analysis… ⏳"):
        if opts["words"]:
            wc, ner, nouns, graph, cloud = wordlevel_pipeline(
                df,
                normalize_language_code(
                    opts["trg_lang" if opts["task"] == "translate" else "src_lang"]
                )
                or "en",
            )
            result["word_counts"] = wc
            result["named_entities"] = ner
            result["noun_sentiment"] = nouns
            result["noun_graph"] = graph
            result["wordcloud"] = cloud

    # Summarization
    with st.spinner("Summarizing… ⏳"):
        if opts["summarization"]:
            if inference_pipeline is None:
                transcript_lang = opts[
                    "trg_lang" if opts["task"] == "translate" else "src_lang"
                ]
                inference_pipeline = InferencePipeline(
                    out_language=_language_name(transcript_lang)
                )
                if not inference_pipeline.get_health():
                    raise ConnectionError(
                        "The configured inference provider is not reachable. Please ensure it is running and accessible."
                    )
            result["summary"] = summarization_pipeline(
                " ".join(df["text"].astype(str).tolist()),
                inference_pipeline=inference_pipeline,
            )

    st.session_state["result"] = result
    st.success("Done! Select another tab to view results.")


def _start_page() -> None:
    """Render the pipeline settings form in the Parameters tab."""
    st.markdown("## ⚙️ Pipeline settings")

    uploaded = st.file_uploader(
        "Audio / video file", type=["wav", "mp3", "m4a", "mp4", "mkv", "webm"]
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

    run = st.button("▶️ Run", disabled=not uploaded)

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

    if run and uploaded is not None:
        # Save upload to a temp file first
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded.name).suffix
        ) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)
        _run_pipeline(tmp_path, st.session_state["opts"])


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
    if "result" not in st.session_state:
        msg = (
            "After you upload a file and press **Run** in the "
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

    result = st.session_state["result"]

    # ---------------- Transcript tab ----------------
    with tab_transcript:
        st.subheader("📜 Transcript")
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
                file_name="summary.txt",
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
