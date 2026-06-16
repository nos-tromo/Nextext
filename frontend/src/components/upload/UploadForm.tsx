import { useState } from 'react'
import { useLanguages } from '../../hooks/useLanguages'
import { checkUploadAcceptable } from '../../lib/uploadGuard'
import { Dropzone } from './Dropzone'
import { ErrorBanner } from '../common/ErrorBanner'
import type { JobOptions, Task } from '../../api/types'

export interface UploadFormProps {
  pending: boolean
  onRun: (files: File[], options: JobOptions) => void
}

/** Pipeline-options form + Dropzone. Calls onRun(files, options) on submit. */
export function UploadForm({ pending, onRun }: UploadFormProps) {
  const languages = useLanguages()
  const [files, setFiles] = useState<File[]>([])
  const [task, setTask] = useState<Task>('transcribe')
  const [srcLang, setSrcLang] = useState<string>('')
  const [trgLang, setTrgLang] = useState<string>('de')
  const [speakers, setSpeakers] = useState<number>(1)
  const [words, setWords] = useState(false)
  const [summarization, setSummarization] = useState(false)
  const [hateSpeech, setHateSpeech] = useState(false)

  const sizeError = checkUploadAcceptable(files)
  const canRun = files.length > 0 && !pending && !sizeError

  function run() {
    if (!canRun) return
    onRun(files, {
      src_lang: srcLang || null,
      trg_lang: trgLang,
      task,
      speakers,
      words,
      summarization,
      hate_speech: hateSpeech,
    })
  }

  const whisper = languages.data?.whisper ?? []
  const target = languages.data?.target ?? []

  return (
    <div className="space-y-4">
      <Dropzone onFiles={(f) => setFiles((prev) => [...prev, ...f])} disabled={pending} />

      {files.length > 0 && (
        <ul className="text-sm text-muted-foreground">
          {files.map((f, i) => (
            <li key={`${f.name}-${i}`} className="flex justify-between">
              <span>{f.name}</span>
              <button className="text-primary" onClick={() => setFiles(files.filter((_, j) => j !== i))} disabled={pending}>
                remove
              </button>
            </li>
          ))}
        </ul>
      )}

      {sizeError && <ErrorBanner message={sizeError} />}

      <div className="grid grid-cols-2 gap-4">
        <label className="space-y-1">
          <span className="text-sm text-muted-foreground">Task</span>
          <select className="w-full rounded border border-border bg-muted px-2 py-1" value={task} onChange={(e) => setTask(e.target.value as Task)}>
            <option value="transcribe">transcribe</option>
            <option value="translate">translate</option>
          </select>
        </label>
        <label className="space-y-1">
          <span className="text-sm text-muted-foreground">Max speakers</span>
          <input type="number" min={1} max={10} className="w-full rounded border border-border bg-muted px-2 py-1" value={speakers} onChange={(e) => setSpeakers(Number(e.target.value))} />
        </label>
        <label className="space-y-1">
          <span className="text-sm text-muted-foreground">Source language</span>
          <select className="w-full rounded border border-border bg-muted px-2 py-1" value={srcLang} onChange={(e) => setSrcLang(e.target.value)}>
            <option value="">Detect language</option>
            {whisper.map((l) => (
              <option key={l.code} value={l.code}>{l.name}</option>
            ))}
          </select>
        </label>
        <label className="space-y-1">
          <span className="text-sm text-muted-foreground">Target language (translate)</span>
          <select className="w-full rounded border border-border bg-muted px-2 py-1" value={trgLang} onChange={(e) => setTrgLang(e.target.value)}>
            {target.map((l) => (
              <option key={l.code} value={l.code}>{l.name}</option>
            ))}
          </select>
        </label>
      </div>

      <div className="flex gap-4 text-sm">
        <label className="flex items-center gap-2"><input type="checkbox" checked={words} onChange={(e) => setWords(e.target.checked)} /> Word analysis</label>
        <label className="flex items-center gap-2"><input type="checkbox" checked={summarization} onChange={(e) => setSummarization(e.target.checked)} /> Summary</label>
        <label className="flex items-center gap-2"><input type="checkbox" checked={hateSpeech} onChange={(e) => setHateSpeech(e.target.checked)} /> Hate speech</label>
      </div>

      <button
        className="rounded bg-primary px-4 py-2 text-primary-foreground disabled:opacity-50"
        disabled={!canRun}
        onClick={run}
      >
        {pending ? 'Submitting…' : '▶ Run'}
      </button>
    </div>
  )
}
