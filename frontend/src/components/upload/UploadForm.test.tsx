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
