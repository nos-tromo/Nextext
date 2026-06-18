import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { LanguagesResponse } from '../../api/types'
import { TARGET_LANG_STORAGE_KEY } from '../../lib/targetLang'
import { UploadForm } from './UploadForm'

const LANGUAGES: LanguagesResponse = {
  whisper: [{ code: 'en', name: 'English' }],
  target: [
    { code: 'ar-EG', name: 'Arabic (Egypt)' },
    { code: 'de-DE', name: 'German (Germany)' },
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
    localStorage.setItem(TARGET_LANG_STORAGE_KEY, 'de-DE')
    mountForm()
    await waitFor(() => expect(targetSelect().value).toBe('de-DE'))
  })
})
