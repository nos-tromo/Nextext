import type { Strings } from './index'

export const de: Strings = {
  // common
  'common.loading': 'Lädt…',
  'common.copied': 'Kopiert',
  'common.cancel': 'Abbrechen',
  'common.delete': 'Löschen',
  'common.close': 'Schließen',
  'common.settings': 'Einstellungen',
  'common.sign_out': 'Abmelden',

  // upload
  'upload.drop_hint': 'Audio- oder Videodateien hier ablegen',
  'upload.select_files': 'Dateien wählen',
  'upload.uploading': 'Wird hochgeladen…',
  'upload.file_too_large': 'Datei ist zu groß',
  'upload.unsupported_format': 'Nicht unterstütztes Dateiformat',
  'upload.read_error': 'Datei konnte nicht gelesen werden',

  // options
  'options.source_language': 'Ausgangssprache',
  'options.target_language': 'Zielsprache',
  'options.auto_detect': 'Automatisch erkennen',
  'options.transcription': 'Transkription',
  'options.translation': 'Übersetzung',
  'options.analyze_entities': 'Entitäten analysieren',
  'options.summarization': 'Zusammenfassung',
  'options.hate_speech_detection': 'Erkennung von Hassrede',
  'options.speaker_diarization': 'Sprechererkennung',

  // processing
  'processing.transcribing': 'Wird transkribiert…',
  'processing.translating': 'Wird übersetzt…',
  'processing.summarizing': 'Wird zusammengefasst…',
  'processing.analyzing_entities': 'Entitäten werden analysiert…',
  'processing.detecting_hate_speech': 'Hassrede wird erkannt…',
  'processing.diarizing': 'Sprecher werden erkannt…',
  'processing.complete': 'Abgeschlossen',
  'processing.failed': 'Fehlgeschlagen',
  'processing.cancelled': 'Abgebrochen',

  // artifacts
  'artifacts.download_transcript': 'Transkription herunterladen',
  'artifacts.download_translation': 'Übersetzung herunterladen',
  'artifacts.download_summary': 'Zusammenfassung herunterladen',
  'artifacts.download_wordcounts': 'Worthäufigkeiten herunterladen',
  'artifacts.download_entities': 'Entitäten herunterladen',
  'artifacts.download_hate_speech': 'Analyse von Hassrede herunterladen',
  'artifacts.download_archive': 'Alles herunterladen',

  // errors
  'errors.upload_failed': 'Hochladen fehlgeschlagen: {error}',
  'errors.processing_failed': 'Verarbeitung fehlgeschlagen: {error}',
  'errors.network_error': 'Netzwerkfehler',
  'errors.try_again': 'Erneut versuchen',
}
