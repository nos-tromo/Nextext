from __future__ import annotations

import tempfile
from pathlib import Path

import altair as alt
import streamlit as st
import streamlit.components.v1 as components

from nextext import pipeline as ne
from nextext.utils import kv_to_vk, load_and_sort_mappings, setup_logging

setup_logging()


def _run_pipeline(tmp_file: Path, opts: dict) -> None:
    """
    Run the core pipeline and stash results in st.session_state.

    Args:
        tmp_file (Path): Path to the temporary file.
        opts (dict): Options for the pipeline.
    """
    # Transcription
    with st.spinner("Transcribing… this might take a while ⏳"):
        df, updated_src_lang = ne.transcription_pipeline(
            file_path=tmp_file,
            src_lang=opts["src_lang"],
            model_id=opts["model_id"],
            task=opts["task"],
            api_key=ne.get_api_key() or "",
            speakers=opts["speakers"],
        )
        # Update the source language in the session state
        opts["src_lang"] = updated_src_lang
        st.session_state["opts"] = opts

    # Translation
    with st.spinner("Translating… this might take a while ⏳"):
        if opts["task"] == "translate" and opts["trg_lang"] != "en":
            df = ne.translation_pipeline(df, opts["trg_lang"])

    # Store the DataFrame and default values in session state
    result = {
        "transcript": df,
        "summary": None,
        "topics": None,
        "word_counts": None,
        "noun_sentiment": None,
        "noun_graph": None,
        "named_entities": None,
        "wordcloud": None,
    }

    # Word-level analysis
    with st.spinner("Running word-level analysis… ⏳"):
        if opts["words"]:
            wc, ner, nouns, graph, cloud = ne.wordlevel_pipeline(
                df,
                opts["trg_lang" if opts["task"] == "translate" else "src_lang"],
            )
            result["word_counts"] = wc
            result["named_entities"] = ner
            result["noun_sentiment"] = nouns
            result["noun_graph"] = graph
            result["wordcloud"] = cloud

    # Topic modelling
    with st.spinner("Running topic modelling… ⏳"):
        if opts["topics"]:
            topics_output = ne.topics_pipeline(
                df,
                opts["trg_lang" if opts["task"] == "translate" else "src_lang"],
            )
            if topics_output is not None:
                result["topics"] = topics_output
            else:
                result["topics"] = None, None

    # Summarization
    with st.spinner("Summarizing… ⏳"):
        if opts["summarization"]:
            result["summary"] = ne.summarization_pipeline(
                " ".join(df["text"].astype(str).tolist()),
                opts["trg_lang" if opts["task"] == "translate" else "src_lang"],
            )

    # Toxicity classification
    with st.spinner("Classifying toxicity… ⏳"):
        if opts["toxicity"]:
            df = ne.toxicity_pipeline(df)
            result["transcript"] = df  # updated with extra column

    st.session_state["result"] = result
    st.success("Done! Select another tab to view results.")


def _start_page() -> None:
    """
    Render the pipeline settings form in the Parameters tab.
    """
    st.markdown("## ⚙️ Pipeline settings")

    uploaded = st.file_uploader(
        "Audio / video file", type=["wav", "mp3", "m4a", "mp4", "mkv", "webm"]
    )

    # Load source language mappings from Whisper and target language mappings from Madlad
    src_lang_maps, src_lang_names = load_and_sort_mappings("whisper_languages.json")
    trg_lang_maps, trg_lang_names = load_and_sort_mappings("madlad_languages.json")

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
        col3, col4 = st.columns(2)
        with col3:
            words = st.checkbox("Word-level analysis")
            topics = st.checkbox("Topic modelling")
        with col4:
            summarization = st.checkbox("Summarisation")
            toxicity = st.checkbox("Toxicity")
    with col2:
        trg_lang_name = st.selectbox(
            "Target language (for translate task)",
            trg_lang_names,
            index=trg_lang_names.index("German"),
        )
        speakers = st.number_input("Max speakers", 1, 10, value=1, step=1)

    src_lang_code = kv_to_vk(src_lang_maps).get(src_lang_name)
    trg_lang_code = kv_to_vk(trg_lang_maps).get(trg_lang_name, "English")

    run = st.button("▶️ Run", disabled=not uploaded)

    # Persist options for the run
    st.session_state["opts"] = dict(
        src_lang=src_lang_code,
        trg_lang=trg_lang_code,
        model_id=model_id,
        task=task,
        speakers=speakers,
        words=words,
        topics=topics,
        summarization=summarization,
        toxicity=toxicity,
    )

    if run and uploaded is not None:
        # Save upload to a temp file first
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded.name).suffix
        ) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)
        _run_pipeline(tmp_path, st.session_state["opts"])


def _main() -> None:
    """
    Main function to run the Streamlit app (tab‑based navigation).
    """
    st.set_page_config(page_title="Nextext", layout="wide")
    st.title("Nextext – Dashboard Report")

    # Top‑level tabs
    tab_params, tab_transcript, tab_summary, tab_words, tab_topics = st.tabs(
        ["Parameters", "Transcript", "Summary", "Word-level Analysis", "Topics"]
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
        for t in (tab_transcript, tab_summary, tab_words, tab_topics):
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
            st.altair_chart(chart, use_container_width=True)

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

    # ---------------- Topics tab --------------------
    with tab_topics:
        if result["topics"] is None:
            st.warning("Topic modelling not requested.")
        else:
            st.subheader("🗂️ Topics")
            for title, topic in result["topics"]:
                st.subheader(title)
                st.write(topic)
                st.divider()
            st.download_button(
                label="Download",
                data="\n\n".join(
                    f"{title}\n{topic}" for (title, topic) in result["topics"]
                ),
                file_name="topics.txt",
                mime="text/plain",
            )

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


if __name__ == "__main__":
    _main()
