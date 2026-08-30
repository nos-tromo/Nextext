import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { LanguagesResponse } from '../../api/types'
import { TARGET_LANG_STORAGE_KEY } from '../../lib/targetLang'
import { UploadForm } from './UploadForm'

const LANGUAGES: LanguagesResponse = {
  whisper: [{ code: 'en', name: 'English' }],
  target: [
    { code: 'ar', name: 'Arabic' },
    { code: 'de', name: 'German' },
    { code: 'en', name: 'English' },
  ],
  default_target: 'en',
}

vi.mock('../../api/meta', () => ({
  getLanguages: vi.fn(async () => LANGUAGES),
}))

function mountForm() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <UploadForm pending={false} onRun={() => {}} />
    </QueryClientProvider>,
  )
}

function targetSelect(): HTMLSelectElement {
  return screen.getByText('Target language (translate)').parentElement!.querySelector('select')!
}

afterEach(() => {
  localStorage.clear()
  vi.restoreAllMocks()
})

describe('UploadForm target language', () => {
  it('defaults to the backend default_target on a fresh browser', async () => {
    mountForm()
    await waitFor(() => expect(targetSelect().value).toBe('en'))
  })

  it('restores the persisted selection across reloads', async () => {
    localStorage.setItem(TARGET_LANG_STORAGE_KEY, 'de')
    mountForm()
    await waitFor(() => expect(targetSelect().value).toBe('de'))
  })
})

function addFiles(container: HTMLElement, files: File[]) {
  const input = container.querySelector('input[type="file"]') as HTMLInputElement
  Object.defineProperty(input, 'files', { value: files, configurable: true })
  fireEvent.change(input)
}

function audio(name: string, size = 1024): File {
  return new File(['x'.repeat(size)], name, { type: 'audio/mpeg' })
}

describe('UploadForm file list', () => {
  it('dedups re-selected files and shows a count summary', () => {
    const { container } = mountForm()
    addFiles(container, [audio('a.mp3'), audio('b.mp3')])
    addFiles(container, [audio('a.mp3')]) // same name+size → duplicate, dropped
    expect(screen.getByText(/2 files/)).toBeInTheDocument()
    expect(screen.getAllByRole('listitem')).toHaveLength(2)
  })

  it('removes a file via the row remove control', () => {
    const { container } = mountForm()
    addFiles(container, [audio('a.mp3'), audio('b.mp3')])
    fireEvent.click(screen.getByRole('button', { name: 'Remove a.mp3' }))
    expect(screen.queryByText('a.mp3')).toBeNull()
    expect(screen.getByText('b.mp3')).toBeInTheDocument()
  })

  it('clears all files via the header action', () => {
    const { container } = mountForm()
    addFiles(container, [audio('a.mp3')])
    fireEvent.click(screen.getByRole('button', { name: 'Clear all' }))
    expect(screen.queryByRole('listitem')).toBeNull()
  })
})

describe('UploadForm option order', () => {
  it('reads in the order the pipeline works', () => {
    // What was heard, then what was seen, then the analyses over that
    // material, and last the summary — which draws on everything before it.
    // Asserted because the order is a decision, not an accident of editing.
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={qc}>
        <UploadForm pending={false} onRun={vi.fn()} />
      </QueryClientProvider>,
    )

    const labels = screen
      .getAllByRole('button')
      .filter((b) => b.hasAttribute('aria-pressed'))
      .map((b) => b.textContent)

    expect(labels).toEqual([
      'Detect speakers',
      'Keyframes',
      'Word analysis',
      'Hate speech',
      'Summary',
    ])
  })
})

describe('UploadForm keyframes toggle', () => {
  it('submits keyframes=false by default and true when ticked', () => {
    // Opt-in, like Summary: sampling and describing frames costs one vision
    // request per frame, so it is never spent unasked.
    const onRun = vi.fn()
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { container } = render(
      <QueryClientProvider client={qc}>
        <UploadForm pending={false} onRun={onRun} />
      </QueryClientProvider>,
    )
    addFiles(container, [audio('a.mp3')])

    const toggle = screen.getByRole('button', { name: 'Keyframes' })
    expect(toggle).toHaveAttribute('aria-pressed', 'false')

    fireEvent.click(screen.getByRole('button', { name: /Run/ }))
    expect(onRun.mock.calls[0][1]).toMatchObject({ keyframes: false, summarization: false })

    fireEvent.click(toggle)
    expect(toggle).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(screen.getByRole('button', { name: /Run/ }))
    // Independent of the summary: ticking one must not tick the other.
    expect(onRun.mock.calls[1][1]).toMatchObject({ keyframes: true, summarization: false })
  })
})

describe('UploadForm diarize toggle', () => {
  it('submits diarize=true by default and false when unchecked', () => {
    const onRun = vi.fn()
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { container } = render(
      <QueryClientProvider client={qc}>
        <UploadForm pending={false} onRun={onRun} />
      </QueryClientProvider>,
    )
    addFiles(container, [audio('a.mp3')])

    const toggle = screen.getByRole('button', { name: 'Detect speakers' })
    expect(toggle).toHaveAttribute('aria-pressed', 'true')

    fireEvent.click(screen.getByRole('button', { name: /Run/ }))
    expect(onRun.mock.calls[0][1]).toMatchObject({ diarize: true })

    fireEvent.click(toggle)
    expect(toggle).toHaveAttribute('aria-pressed', 'false')
    fireEvent.click(screen.getByRole('button', { name: /Run/ }))
    expect(onRun.mock.calls[1][1]).toMatchObject({ diarize: false })
  })
})
