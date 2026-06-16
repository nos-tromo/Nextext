import { useRef, useState, type DragEvent } from 'react'
import { cn } from '../../lib/cn'

const ACCEPT = '.mp3,.m4a,.mp4,.mkv,.ogg,.wav,.webm'

export interface DropzoneProps {
  onFiles: (files: File[]) => void
  disabled?: boolean
}

/** Click-or-drag file picker. Calls onFiles with the selected File[]. */
export function Dropzone({ onFiles, disabled = false }: DropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  function pick(list: FileList | null) {
    if (!list || list.length === 0) return
    onFiles(Array.from(list))
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setDragging(false)
    if (disabled) return
    pick(e.dataTransfer.files)
  }

  return (
    <div
      role="button"
      tabIndex={0}
      aria-disabled={disabled}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(e) => {
        if (!disabled && (e.key === 'Enter' || e.key === ' ')) inputRef.current?.click()
      }}
      onDragOver={(e) => {
        e.preventDefault()
        if (!disabled) setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={cn(
        'flex cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed px-6 py-10 text-center transition',
        dragging ? 'border-primary bg-primary/10' : 'border-border bg-muted/30',
        disabled && 'cursor-not-allowed opacity-50',
      )}
    >
      <p className="text-foreground">Drop audio/video files here, or click to choose</p>
      <p className="mt-1 text-sm text-muted-foreground">mp3, m4a, mp4, mkv, ogg, wav, webm</p>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={ACCEPT}
        className="hidden"
        disabled={disabled}
        onChange={(e) => {
          pick(e.target.files)
          e.target.value = ''
        }}
      />
    </div>
  )
}
