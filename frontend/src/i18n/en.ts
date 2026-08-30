export const en = {
  // common
  'common.cancel': 'Cancel',
  'common.remove': 'Remove',
  'common.clear_all': 'Clear all',
  'common.error_request': 'Something went wrong ({status}). Please try again or contact support.',
  'common.error_unknown': 'Something went wrong. Please try again or contact support.',
  'common.error_network': 'Service unreachable. Please check your connection or try again later.',
  'common.file_count_one': '{count} file',
  'common.file_count_other': '{count} files',
  'common.jobs_processing': '{count} processing',
  'common.jobs_queued': '{count} queued',
  'common.jobs_finished': '{count} finished',
  'common.jobs_skipped': '{count} skipped',
  'common.jobs_failed': '{count} failed',
  'common.batch_progress': '{pct}% of files done',

  // header (AppHeader)
  'header.home': 'Apps',
  'header.theme_system': 'system',
  'header.theme_light': 'light',
  'header.theme_dark': 'dark',
  'header.sign_out': 'Sign out',

  // page
  'page.title': 'Audio & Video Analysis',
  'page.caption': 'Transcription, keyframe extraction, translation, and analysis',

  // home
  'home.new_job': 'New job',
  'home.jobs_heading': 'Jobs',

  // upload
  'upload.drop_hint': 'Upload audio/video files',
  'upload.file_too_large':
    '"{name}" is {sizeGb} GB, over the {limitGb} GB per-file limit. Split it, or use `nextext-cli` for very large local files.',
  'upload.run': '▶ Run',
  'upload.submitting': 'Submitting…',

  // options
  'options.task': 'Task',
  'options.task_transcribe': 'Transcribe',
  'options.task_translate': 'Transcribe & translate',
  'options.source_language': 'Source language (data)',
  'options.target_language_translate': 'Target language (translate)',
  'options.auto_detect': 'Detect language',
  'options.summary': 'Summary',
  'options.hate_speech': 'Hate speech',
  'options.detect_speakers': 'Detect speakers',
  'options.word_analysis': 'Word analysis',
  'options.keyframes': 'Keyframes',

  // processing (terminal job states only; in-flight stage text is protocol —
  // sent verbatim by the backend as `stage` and never translated client-side)
  'processing.complete': 'Complete',
  'processing.failed': 'Failed',
  'processing.cancelled': 'Cancelled',

  // artifacts
  'artifacts.download_archive': 'Download all (.zip)',

  // downloads (per-artifact button labels; CSV/XLSX/TXT/PNG/JSONL are format
  // abbreviations, identical in both locales, and stay as plain literals)
  'downloads.transcript_txt': 'Transcript TXT',
  'downloads.translation_txt': 'Translation TXT',

  // results (ResultPanel and its tabs)
  // Accessible name for a download button whose visible chip is the bare format.
  'results.download_artifact': 'Download {label}',
  'results.tab_transcript': 'Transcript',
  'results.tab_visual_context': 'Visual context',
  'results.tab_summary': 'Summary',
  'results.tab_words': 'Words',
  'results.tab_wordcloud': 'Word Cloud',
  'results.tab_entities': 'Entities',
  'results.tab_hate_speech': 'Hate Speech',
  'results.tab_nav_label': 'Result tabs',
  'results.loading': 'Loading results…',
  'results.load_failed': 'Failed to load results.',
  'results.no_data': 'No result data available.',
  'results.skipped': 'Job was skipped.',
  'results.skipped_reason': 'Job was skipped: {reason}',
  'results.skipped_title': 'Nothing was transcribed',
  'results.skipped_vad_no_speech':
    'No speech was detected in this file, so it was not transcribed or analyzed.',
  'results.skipped_asr_empty':
    'Transcription returned no text for this file, so there was nothing to analyze.',
  'results.skipped_asr_filtered':
    'Only non-speech audio (music, noise or silence) was detected, so the transcribed text was discarded.',
  'results.no_transcript': 'No transcript segments were produced for this job.',
  'results.no_word_counts': 'No word counts available for this job.',
  'results.col_word': 'Word',
  'results.col_count': 'Count',
  'results.no_entities': 'No named entities found for this job.',
  'results.col_entity': 'Entity',
  'results.col_category': 'Category',
  'results.col_frequency': 'Frequency',
  'player.title': 'Media player',
  'player.close': 'Close player',
  'player.open': 'Open player',
  'player.play_from': 'Play from {time}',
  'player.unplayable': 'This file can’t be played in the browser. Download it to play it locally.',
  'results.no_summary': 'No summary produced for this job.',
  'results.no_visual_context': 'No visual context produced for this job.',
  'results.visual_context_hint': 'What the video showed at each sampled moment. A summary of this job draws on these descriptions too, when one was requested.',
  'results.keyframes_only': 'The keyframes were sampled but not described. Download them to see them.',
  'artifacts.download_keyframes': 'Download the sampled keyframes as a ZIP archive',
  'results.col_time': 'Time',
  'results.col_description': 'What is shown',
  'results.no_hate_speech': 'No hate-speech findings for this job.',
  'results.flagged_summary_one': '{flagged} of {total} segment flagged.',
  'results.flagged_summary_other': '{flagged} of {total} segments flagged.',
  'results.flagged': 'Flagged',
  'results.clean': 'Clean',
  'results.col_start': 'Start',
  'results.col_end': 'End',
  'results.col_speaker': 'Speaker',
  'results.col_transcript': 'Transcript',
  'results.col_text': 'Text',
  'results.col_translation': 'Translation',
  'results.wordcloud_loading': 'Loading word cloud…',
  'results.wordcloud_alt': 'Word cloud',

  // jobs (JobCard, BatchProgress, BatchDownloadMenu, ClearJobsMenu)
  'jobs.status_queued': 'Queued',
  'jobs.status_running': 'Processing',
  'jobs.show_results': 'Show results',
  'jobs.hide_results': 'Hide results',
  'jobs.removing': 'Removing…',
  'jobs.interrupted': 'Job was interrupted before it could finish.',
  'jobs.unknown_error': 'Unknown error',
  'jobs.skipped': 'Skipped — no processable content',
  'jobs.skipped_vad_no_speech': 'Skipped — no speech detected in the audio',
  'jobs.skipped_asr_empty': 'Skipped — transcription returned no text',
  'jobs.skipped_asr_filtered': 'Skipped — only non-speech audio was detected',
  'jobs.error_undecodable': 'File could not be decoded — is it a valid audio or video file?',
  'jobs.waiting': 'Waiting…',
  'jobs.done': 'Done',
  'jobs.stage_progress': '{stage} ({pct}%)',
  'jobs.remove_failed': 'Could not remove job.',
  'jobs.loading': 'Loading jobs…',
  'jobs.load_failed': 'Could not load jobs.',
  'jobs.none_yet': 'No jobs yet.',
  'jobs.download_all': 'Download all jobs',
  'jobs.downloading': 'Downloading…',
  'jobs.no_completed_yet': 'No completed jobs yet',
  'jobs.combined_jsonl': 'Combined JSONL (docint)',
  'jobs.full_batch_zip': 'Full batch (ZIP)',
  'jobs.clear': 'Clear jobs',
  'jobs.clearing': 'Clearing…',
  'jobs.clear_finished': 'Clear finished ({count})',
  'jobs.clear_all': 'Clear all ({count})',
  'jobs.clear_confirm_one': "Remove {count} job? This can't be undone.",
  'jobs.clear_confirm_other': "Remove {count} jobs? This can't be undone.",
  'jobs.clear_confirm_button': 'Clear',
  'jobs.clear_partial_failure': 'Cleared {cleared} of {total}; {failed} failed',
  'jobs.no_jobs_to_clear': 'No jobs to clear',

  // errors
  'errors.upload_failed': 'Submission failed.',
}
