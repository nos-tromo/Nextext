import { useState } from 'react'
import { useLanguages } from '../../hooks/useLanguages'
import { checkUploadAcceptable } from '../../lib/uploadGuard'
import { readStoredTargetLang, writeStoredTargetLang } from '../../lib/targetLang'
import { Dropzone } from './Dropzone'
import { Banner, Button, FileList, Select, ToggleButton, mergeFiles } from '@infra/ui'
import { useT } from '../../i18n/LanguageContext'
import type { JobOptions, Task } from '../../api/types'

export interface UploadFormProps {
  pending: boolean
  onRun: (files: File[], options: JobOptions) => void
}

/** Pipeline-options form + Dropzone. Calls onRun(files, options) on submit. */
export function UploadForm({ pending, onRun }: UploadFormProps) {
  const t = useT()
  const languages = useLanguages()
  const [files, setFiles] = useState<File[]>([])
  const [task, setTask] = useState<Task>('transcribe')
  const [srcLang, setSrcLang] = useState<string>('')
  // `null` means "no active user choice yet" — fall back to the backend default
  // below. A persisted preference (survives reloads) seeds it on mount.
  const [trgLang, setTrgLang] = useState<string | null>(() => readStoredTargetLang())
  const [diarize, setDiarize] = useState<boolean>(true)
  const [words, setWords] = useState(false)
  const [summarization, setSummarization] = useState(false)
  const [hateSpeech, setHateSpeech] = useState(false)
  const [keyframes, setKeyframes] = useState(false)

  const sizeError = checkUploadAcceptable(files, undefined, (vars) =>
    t('upload.file_too_large', { name: vars.name, sizeGb: vars.sizeGb, limitGb: vars.limitGb }),
  )
  const canRun = files.length > 0 && !pending && !sizeError

  function run() {
    if (!canRun) return
    onRun(files, {
      src_lang: srcLang || null,
      trg_lang: effectiveTrgLang,
      task,
      diarize,
      words,
      summarization,
      hate_speech: hateSpeech,
      keyframes,
    })
  }

  const whisper = languages.data?.whisper ?? []
  const target = languages.data?.target ?? []
  const defaultTarget = languages.data?.default_target ?? ''

  // Effective selection: the active/persisted choice while it is a supported
  // code, otherwise the backend default. Derived (not stored) so a stale
  // persisted code or the initial load resolves without an effect.
  const trgValid = trgLang !== null && target.some((l) => l.code === trgLang)
  const effectiveTrgLang = trgValid ? (trgLang as string) : defaultTarget

  function selectTrgLang(code: string) {
    setTrgLang(code)
    writeStoredTargetLang(code)
  }

  return (
    <div className="space-y-4">
      <Dropzone onFiles={(f) => setFiles((prev) => mergeFiles(prev, f))} disabled={pending} />

      <FileList
        files={files}
        onRemove={pending ? undefined : (i) => setFiles(files.filter((_, j) => j !== i))}
        onClear={pending ? undefined : () => setFiles([])}
        labels={{
          clearAll: t('common.clear_all'),
          remove: t('common.remove'),
          files: (count) => t(count === 1 ? 'common.file_count_one' : 'common.file_count_other', { count }),
        }}
      />

      {sizeError && <Banner variant="danger">{sizeError}</Banner>}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <label className="space-y-1">
          <span className="text-sm text-muted-foreground">{t('options.task')}</span>
          <Select className="w-full" value={task} onChange={(e) => setTask(e.target.value as Task)}>
            <option value="transcribe">{t('options.task_transcribe')}</option>
            <option value="translate">{t('options.task_translate')}</option>
          </Select>
        </label>
        <label className="space-y-1">
          <span className="text-sm text-muted-foreground">{t('options.source_language')}</span>
          <Select className="w-full" value={srcLang} onChange={(e) => setSrcLang(e.target.value)}>
            <option value="">{t('options.auto_detect')}</option>
            {whisper.map((l) => (
              <option key={l.code} value={l.code}>{l.name}</option>
            ))}
          </Select>
        </label>
        <label className="space-y-1">
          <span className="text-sm text-muted-foreground">{t('options.target_language_translate')}</span>
          <Select className="w-full" value={effectiveTrgLang} onChange={(e) => selectTrgLang(e.target.value)}>
            {target.map((l) => (
              <option key={l.code} value={l.code}>{l.name}</option>
            ))}
          </Select>
        </label>
      </div>

      {/* The options fill the form's width like the Run button under them, so
          which stages a run includes is read as a row of lit panels rather
          than hunted for in five small boxes. `flex-1` shares the span
          evenly; the minimum width is what makes them wrap instead of crush
          on a narrow screen or in a long-worded locale. */}
      <div className="flex flex-wrap gap-2">
        <ToggleButton className="min-w-32 flex-1" pressed={diarize} onClick={() => setDiarize((v) => !v)}>
          {t('options.detect_speakers')}
        </ToggleButton>
        <ToggleButton className="min-w-32 flex-1" pressed={words} onClick={() => setWords((v) => !v)}>
          {t('options.word_analysis')}
        </ToggleButton>
        <ToggleButton className="min-w-32 flex-1" pressed={summarization} onClick={() => setSummarization((v) => !v)}>
          {t('options.summary')}
        </ToggleButton>
        <ToggleButton className="min-w-32 flex-1" pressed={hateSpeech} onClick={() => setHateSpeech((v) => !v)}>
          {t('options.hate_speech')}
        </ToggleButton>
        <ToggleButton className="min-w-32 flex-1" pressed={keyframes} onClick={() => setKeyframes((v) => !v)}>
          {t('options.keyframes')}
        </ToggleButton>
      </div>

      <Button type="button" className="w-full" disabled={!canRun} onClick={run}>
        {pending ? t('upload.submitting') : t('upload.run')}
      </Button>
    </div>
  )
}
