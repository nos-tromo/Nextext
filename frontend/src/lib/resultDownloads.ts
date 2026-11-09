import type { DownloadSpec } from '../components/results/DownloadButtons'
import type { JobResult } from '../api/types'
import type { useT } from '../i18n/LanguageContext'
import { transcriptHasTranslation } from './transcriptTable'

/** The result panel's tabs. */
export type ResultTabId =
  | 'transcript'
  | 'visual_context'
  | 'faces'
  | 'summary'
  | 'words'
  | 'wordcloud'
  | 'entities'
  | 'hate_speech'

type Translate = ReturnType<typeof useT>

/** A tab's CSV and XLSX pair, which most tables offer. */
function table(artifact: string, stem: string): DownloadSpec[] {
  return [
    { name: `${artifact}.csv`, label: 'CSV', fileName: `${stem}_${artifact}.csv` },
    { name: `${artifact}.xlsx`, label: 'XLSX', fileName: `${stem}_${artifact}.xlsx` },
  ]
}

/**
 * The downloads a result tab offers, in the order its header shows them.
 *
 * One list for the whole panel, so the header can show the active tab's
 * formats beside play and the archive instead of each tab hanging its own row
 * of buttons under its content.
 *
 * @param tab - The active tab.
 * @param result - The job's result.
 * @param stem - The upload's name without its extension, prefixing every file.
 * @param t - The translator, for the labels that are words rather than formats.
 * @returns The tab's downloads; empty for a tab without any.
 */
export function downloadsFor(tab: ResultTabId, result: JobResult, stem: string, t: Translate): DownloadSpec[] {
  const keyframes: DownloadSpec[] = result.keyframes_url
    ? [
        {
          name: 'keyframes.zip',
          label: t('downloads.keyframes_zip'),
          title: t('artifacts.download_keyframes'),
          fileName: `${stem}_keyframes.zip`,
        },
      ]
    : []
  switch (tab) {
    case 'transcript':
      // Nothing to export from an empty transcript; the backend has no such files.
      if (result.transcript.length === 0) return []
      return [
        ...table('transcript', stem),
        ...(transcriptHasTranslation(result.transcript)
          ? [
              { name: 'transcript.txt', label: t('downloads.transcript_txt'), fileName: `${stem}_transcript.txt` },
              { name: 'translation.txt', label: t('downloads.translation_txt'), fileName: `${stem}_translation.txt` },
            ]
          : [{ name: 'transcript.txt', label: 'TXT', fileName: `${stem}_transcript.txt` }]),
        { name: 'docint.jsonl', label: 'JSONL', fileName: `${stem}_docint.jsonl` },
      ]
    case 'visual_context':
      return [
        ...((result.frame_captions ?? []).length > 0
          ? [{ name: 'visual_context.txt', label: 'TXT', fileName: `${stem}_visual_context.txt` }]
          : []),
        ...keyframes,
      ]
    case 'faces':
      // The backend has no faces.zip or faces.csv without a face to put in them.
      if (!result.faces || result.faces.outcome === 'unavailable' || result.faces.entries.length === 0) return []
      return [
        { name: 'faces.zip', label: 'ZIP', title: t('artifacts.download_faces'), fileName: `${stem}_faces.zip` },
        { name: 'faces.csv', label: 'CSV', title: t('artifacts.download_faces_csv'), fileName: `${stem}_faces.csv` },
      ]
    case 'summary':
      return [{ name: 'summary.txt', label: 'TXT', fileName: `${stem}_summary.txt` }]
    case 'words':
      return table('wordcounts', stem)
    case 'wordcloud':
      return [{ name: 'wordcloud.png', label: 'PNG', fileName: `${stem}_wordcloud.png` }]
    case 'entities':
      return table('entities', stem)
    case 'hate_speech':
      return table('hate_speech', stem)
  }
}
