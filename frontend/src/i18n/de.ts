import type { Strings } from './index'

export const de: Strings = {
  // common
  'common.cancel': 'Abbrechen',
  'common.remove': 'Entfernen',
  'common.clear_all': 'Alle entfernen',
  'common.error_request': 'Etwas ist schiefgelaufen ({status}) – Bitte versuchen Sie es erneut oder kontaktieren Sie den Support.',
  'common.error_unknown': 'Etwas ist schiefgelaufen – Bitte versuchen Sie es erneut oder kontaktieren Sie den Support.',
  'common.error_network': 'Dienst nicht erreichbar – Bitte prüfen Sie Ihre Verbindung oder versuchen Sie es später erneut.',
  'common.file_count_one': '{count} Datei',
  'common.file_count_other': '{count} Dateien',
  'common.jobs_processing': '{count} in Verarbeitung',
  'common.jobs_queued': '{count} in Warteschlange',
  'common.jobs_finished': '{count} abgeschlossen',
  'common.jobs_skipped': '{count} übersprungen',
  'common.jobs_failed': '{count} fehlgeschlagen',
  'common.batch_progress': '{pct}% der Dateien fertig',

  // header (AppHeader)
  'header.home': 'Übersicht',
  'header.theme_system': 'System',
  'header.theme_light': 'Hell',
  'header.theme_dark': 'Dunkel',
  'header.sign_out': 'Abmelden',

  // page
  'page.title': 'Audio- und Videoanalyse',
  'page.caption': 'Transkription, Keyframe-Extraktion, Übersetzung und Analyse',

  // home
  'home.new_job': 'Neuer Auftrag',
  'home.jobs_heading': 'Aufträge',

  // upload
  'upload.drop_hint': 'Audio- oder Videodateien hochladen',
  'upload.file_too_large':
    '"{name}" ist {sizeGb} GB groß und überschreitet das Limit von {limitGb} GB pro Datei. Teilen Sie die Datei auf, oder verwenden Sie `nextext-cli` für sehr große lokale Dateien.',
  'upload.run': '▶ Starten',
  'upload.submitting': 'Wird übermittelt…',

  // options
  'options.task': 'Aufgabe',
  'options.task_transcribe': 'Transkribieren',
  'options.task_translate': 'Transkribieren & Übersetzen',
  'options.source_language': 'Ausgangssprache (Daten)',
  'options.target_language_translate': 'Zielsprache (Übersetzung)',
  'options.auto_detect': 'Sprache erkennen',
  'options.summary': 'Zusammenfassung',
  'options.hate_speech': 'Hatespeech',
  'options.detect_speakers': 'Sprechererkennung',
  'options.word_analysis': 'Wortanalyse',

  // processing (terminal job states only; in-flight stage text is protocol —
  // sent verbatim by the backend as `stage` and never translated client-side)
  'processing.complete': 'Abgeschlossen',
  'processing.failed': 'Fehlgeschlagen',
  'processing.cancelled': 'Abgebrochen',

  // artifacts
  'artifacts.download_archive': 'Alles herunterladen (.zip)',

  // downloads
  'downloads.transcript_txt': 'Transkript TXT',
  'downloads.translation_txt': 'Übersetzung TXT',

  // results (ResultPanel and its tabs)
  // Barrierefreier Name eines Download-Buttons, dessen Chip nur das Format zeigt.
  'results.download_artifact': '{label} herunterladen',
  'results.tab_transcript': 'Transkript',
  'results.tab_visual_context': 'Visueller Kontext',
  'results.tab_summary': 'Zusammenfassung',
  'results.tab_words': 'Wörter',
  'results.tab_wordcloud': 'Wörterwolke',
  'results.tab_entities': 'Entitäten',
  'results.tab_hate_speech': 'Hatespeech',
  'results.tab_nav_label': 'Ergebnis-Tabs',
  'results.loading': 'Ergebnisse werden geladen…',
  'results.load_failed': 'Ergebnisse konnten nicht geladen werden.',
  'results.no_data': 'Keine Ergebnisdaten verfügbar.',
  'results.skipped': 'Auftrag wurde übersprungen.',
  'results.skipped_reason': 'Auftrag wurde übersprungen: {reason}',
  'results.skipped_title': 'Es wurde nichts transkribiert',
  'results.skipped_vad_no_speech':
    'In dieser Datei wurde keine Sprache erkannt, daher wurde sie weder transkribiert noch analysiert.',
  'results.skipped_asr_empty':
    'Die Transkription hat für diese Datei keinen Text geliefert, daher gab es nichts zu analysieren.',
  'results.skipped_asr_filtered':
    'Es wurde ausschließlich Nicht-Sprache (Musik, Geräusche oder Stille) erkannt, daher wurde der transkribierte Text verworfen.',
  'results.no_transcript': 'Für diesen Auftrag wurden keine Transkriptsegmente erzeugt.',
  'results.no_word_counts': 'Keine Worthäufigkeiten für diesen Auftrag verfügbar.',
  'results.col_word': 'Wort',
  'results.col_count': 'Anzahl',
  'results.no_entities': 'Keine Entitäten für diesen Auftrag gefunden.',
  'results.col_entity': 'Entität',
  'results.col_category': 'Kategorie',
  'results.col_frequency': 'Häufigkeit',
  'results.no_summary': 'Keine Zusammenfassung für diesen Auftrag erstellt.',
  'results.no_visual_context': 'Kein visueller Kontext für diesen Auftrag erstellt.',
  'results.visual_context_hint': 'Was das Video zu den jeweils ausgewerteten Zeitpunkten zeigte. Eine Zusammenfassung dieses Auftrags berücksichtigt diese Beschreibungen ebenfalls.',
  'results.col_time': 'Zeit',
  'results.col_description': 'Zu sehen',
  'results.no_hate_speech': 'Keine Hatespeech-Befunde für diesen Auftrag.',
  'results.flagged_summary_one': '{flagged} von {total} Segment markiert.',
  'results.flagged_summary_other': '{flagged} von {total} Segmenten markiert.',
  'results.flagged': 'Markiert',
  'results.clean': 'Unauffällig',
  'results.col_start': 'Start',
  'results.col_end': 'Ende',
  'results.col_speaker': 'Sprecher',
  'results.col_transcript': 'Transkript',
  'results.col_text': 'Text',
  'results.col_translation': 'Übersetzung',
  'results.wordcloud_loading': 'Wörterwolke wird geladen…',
  'results.wordcloud_alt': 'Wörterwolke',

  // jobs (JobCard, BatchProgress, BatchDownloadMenu, ClearJobsMenu)
  'jobs.status_queued': 'In Warteschlange',
  'jobs.status_running': 'Läuft',
  'jobs.show_results': 'Ergebnisse anzeigen',
  'jobs.hide_results': 'Ergebnisse ausblenden',
  'jobs.removing': 'Wird entfernt…',
  'jobs.interrupted': 'Der Auftrag wurde unterbrochen, bevor er abgeschlossen werden konnte.',
  'jobs.unknown_error': 'Unbekannter Fehler',
  'jobs.skipped': 'Übersprungen — kein verarbeitbarer Inhalt',
  'jobs.skipped_vad_no_speech': 'Übersprungen — keine Sprache im Audio erkannt',
  'jobs.skipped_asr_empty': 'Übersprungen — Transkription lieferte keinen Text',
  'jobs.skipped_asr_filtered': 'Übersprungen — nur Nicht-Sprache erkannt',
  'jobs.error_undecodable': 'Datei konnte nicht dekodiert werden — ist es eine gültige Audio- oder Videodatei?',
  'jobs.waiting': 'Wartet…',
  'jobs.done': 'Fertig',
  'jobs.stage_progress': '{stage} ({pct}%)',
  'jobs.remove_failed': 'Auftrag konnte nicht entfernt werden.',
  'jobs.loading': 'Aufträge werden geladen…',
  'jobs.load_failed': 'Aufträge konnten nicht geladen werden.',
  'jobs.none_yet': 'Noch keine Aufträge.',
  'jobs.download_all': 'Alle Aufträge herunterladen',
  'jobs.downloading': 'Wird heruntergeladen…',
  'jobs.no_completed_yet': 'Noch keine abgeschlossenen Aufträge',
  'jobs.combined_jsonl': 'Kombinierte JSONL (docint)',
  'jobs.full_batch_zip': 'Gesamter Batch (ZIP)',
  'jobs.clear': 'Aufträge löschen',
  'jobs.clearing': 'Wird gelöscht…',
  'jobs.clear_finished': 'Abgeschlossene löschen ({count})',
  'jobs.clear_all': 'Alle löschen ({count})',
  'jobs.clear_confirm_one': '{count} Auftrag entfernen? Dies kann nicht rückgängig gemacht werden.',
  'jobs.clear_confirm_other': '{count} Aufträge entfernen? Dies kann nicht rückgängig gemacht werden.',
  'jobs.clear_confirm_button': 'Löschen',
  'jobs.clear_partial_failure': '{cleared} von {total} gelöscht; {failed} fehlgeschlagen',
  'jobs.no_jobs_to_clear': 'Keine Aufträge zu löschen',

  // errors
  'errors.upload_failed': 'Übermittlung fehlgeschlagen.',
}
