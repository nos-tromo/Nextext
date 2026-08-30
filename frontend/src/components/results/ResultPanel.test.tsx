import { afterEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { JobSnapshot } from '../../api/types'
import { ResultPanel } from './ResultPanel'
import { useMediaPlayerStore } from '../../lib/mediaPlayerStore'

function mountResultPanel(jobId: string, fileName: string) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <ResultPanel jobId={jobId} fileName={fileName} />
    </QueryClientProvider>,
  )
}

/** Minimal completed JobSnapshot with a transcript and no optional data. */
function makeSnapshot(overrides: Partial<JobSnapshot['result']> = {}): JobSnapshot {
  return {
    job_id: 'j1',
    status: 'completed',
    file_name: 'clip.wav',
    source_file_hash: null,
    options: {
      src_lang: null,
      trg_lang: 'en',
      task: 'transcribe',
      diarize: true,
      words: false,
      summarization: false,
      hate_speech: false,
  keyframes: false,
    },
    stage: null,
    stage_index: 3,
    progress: 1,
    error: null,
    error_code: null,
    skipped: false,
    skip_reason_code: null,
    created_at: '2026-01-01T00:00:00Z',
    started_at: '2026-01-01T00:00:01Z',
    finished_at: '2026-01-01T00:01:00Z',
    result: {
      transcript: [
        {
          start: '0.00',
          end: '2.00',
          start_seconds: 0,
          end_seconds: 2,
          speaker: null,
          text: 'Hello world',
          translation: null,
        },
      ],
      transcript_language: 'en',
      resolved_src_lang: 'en',
      summary: null,
      word_counts: null,
      named_entities: null,
      wordcloud_url: null,
      keyframes_url: null,
      media_url: null,
      frame_captions: null,
      hate_speech_findings: null,
      skipped: false,
      skip_reason: null,
      skip_reason_code: null,
      task: 'transcribe',
      ...overrides,
    },
  }
}

afterEach(() => {
  vi.restoreAllMocks()
  useMediaPlayerStore.getState().clear()
})

describe('ResultPanel', () => {
  it('renders the transcript tab and its text after data loads', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify(makeSnapshot()), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
      ),
    )

    mountResultPanel('j1', 'clip.wav')

    await waitFor(() => expect(screen.getByText('Hello world')).toBeInTheDocument())

    // Transcript tab button should be visible
    expect(screen.getByRole('button', { name: 'Transcript' })).toBeInTheDocument()
  })

  it('does not render the Summary tab when summary is absent', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify(makeSnapshot({ summary: null })), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
      ),
    )

    mountResultPanel('j1', 'clip.wav')

    await waitFor(() => expect(screen.getByText('Hello world')).toBeInTheDocument())

    expect(screen.queryByRole('button', { name: 'Summary' })).not.toBeInTheDocument()
  })

  it('renders the Summary tab when a summary is present', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(
          JSON.stringify(makeSnapshot({ summary: 'A brief summary.' })),
          { status: 200, headers: { 'content-type': 'application/json' } },
        ),
      ),
    )

    mountResultPanel('j1', 'clip.wav')

    await waitFor(() => expect(screen.getByRole('button', { name: 'Summary' })).toBeInTheDocument())
  })

  it('does not render the Visual context tab for an audio-only job', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify(makeSnapshot({ frame_captions: null })), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
      ),
    )

    mountResultPanel('j1', 'clip.wav')

    await waitFor(() => expect(screen.getByText('Hello world')).toBeInTheDocument())
    expect(screen.queryByRole('button', { name: 'Visual context' })).not.toBeInTheDocument()
  })

  it('places the Visual context tab directly after Transcript', async () => {
    // It describes the same material the transcript covers, moment by moment,
    // so it belongs beside it rather than after the derived analyses.
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(
          JSON.stringify(
            makeSnapshot({
              summary: 'A brief summary.',
              frame_captions: [{ time_sec: 0, caption: 'A slide titled Roadmap' }],
            }),
          ),
          { status: 200, headers: { 'content-type': 'application/json' } },
        ),
      ),
    )

    mountResultPanel('j1', 'clip.wav')

    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Visual context' })).toBeInTheDocument(),
    )
    const tabs = screen.getByRole('navigation').querySelectorAll('button')
    expect([...tabs].map((b) => b.textContent)).toEqual(['Transcript', 'Visual context', 'Summary'])
  })

  it('shows the captions when the Visual context tab is selected', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(
          JSON.stringify(
            makeSnapshot({ frame_captions: [{ time_sec: 5, caption: 'A speaker at a lectern' }] }),
          ),
          { status: 200, headers: { 'content-type': 'application/json' } },
        ),
      ),
    )

    mountResultPanel('j1', 'clip.wav')

    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Visual context' })).toBeInTheDocument(),
    )
    fireEvent.click(screen.getByRole('button', { name: 'Visual context' }))
    expect(screen.getByText('A speaker at a lectern')).toBeInTheDocument()
    expect(screen.getByText('00:05')).toBeInTheDocument()
  })

  it('explains a skipped job in a banner, localized from the typed code', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(
          JSON.stringify(
            makeSnapshot({
              transcript: [],
              skipped: true,
              // Backend prose the UI must NOT render: the SPA localizes the code.
              skip_reason: 'No speech detected in the audio.',
              skip_reason_code: 'vad_no_speech',
            }),
          ),
          { status: 200, headers: { 'content-type': 'application/json' } },
        ),
      ),
    )

    mountResultPanel('j1', 'clip.wav')

    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument())
    expect(screen.getByText(/No speech was detected in this file/)).toBeInTheDocument()
    expect(screen.queryByText('No speech detected in the audio.')).not.toBeInTheDocument()
  })

  it('keeps a skipped job\u2019s visual results under the banner', async () => {
    // "Skipped" means no transcript, not an empty job: a silent video still
    // has keyframes, their descriptions, and a summary written from them.
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(
          JSON.stringify(
            makeSnapshot({
              transcript: [],
              skipped: true,
              skip_reason_code: 'vad_no_speech',
              frame_captions: [{ time_sec: 0, caption: 'An empty street' }],
              keyframes_url: '/api/v1/jobs/j1/artifacts/keyframes.zip',
              summary: 'A quiet street scene.',
            }),
          ),
          { status: 200, headers: { 'content-type': 'application/json' } },
        ),
      ),
    )

    mountResultPanel('j1', 'clip.mp4')

    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument())
    expect(screen.getByRole('button', { name: 'Visual context' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Summary' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Transcript' })).not.toBeInTheDocument()
  })

  it('shows the Visual context tab for frames that were never described', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(
          JSON.stringify(
            makeSnapshot({ frame_captions: null, keyframes_url: '/api/v1/jobs/j1/artifacts/keyframes.zip' }),
          ),
          { status: 200, headers: { 'content-type': 'application/json' } },
        ),
      ),
    )

    mountResultPanel('j1', 'clip.mp4')

    await waitFor(() => expect(screen.getByRole('button', { name: 'Visual context' })).toBeInTheDocument())
  })

  it('falls back to the generic skipped message when no code is present', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(
          JSON.stringify(makeSnapshot({ transcript: [], skipped: true, skip_reason_code: null })),
          { status: 200, headers: { 'content-type': 'application/json' } },
        ),
      ),
    )

    mountResultPanel('j1', 'clip.wav')

    await waitFor(() => expect(screen.getByText('Job was skipped.')).toBeInTheDocument())
  })

  it('shows an empty state rather than a headers-only table for an empty transcript', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify(makeSnapshot({ transcript: [] })), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
      ),
    )

    mountResultPanel('j1', 'clip.wav')

    await waitFor(() => expect(screen.getByText(/No transcript segments/)).toBeInTheDocument())
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })

  it('shows spinner while loading', () => {
    // Never resolves — keeps the component in loading state
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})))

    mountResultPanel('j1', 'clip.wav')

    expect(screen.getByRole('status')).toBeInTheDocument()
  })

  it('shows error banner on fetch failure', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ detail: 'Not found' }), {
          status: 404,
          headers: { 'content-type': 'application/json' },
        }),
      ),
    )

    mountResultPanel('j1', 'clip.wav')

    await waitFor(() =>
      expect(screen.getByText(/Failed to load results/)).toBeInTheDocument(),
    )
  })

  it('renders the archive download button with stem-prefixed filename', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify(makeSnapshot()), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
      ),
    )

    const screenApi = mountResultPanel('j1', 'clip.wav')

    await waitFor(() => {
      const btn = screenApi.getByRole('button', { name: 'Download all (.zip)' })
      expect(btn).toBeInTheDocument()
    })
  })
})

describe('ResultPanel media player', () => {
  it('offers an Open player control when the recording is still available', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify(makeSnapshot({ media_url: '/api/v1/jobs/j1/media?token=t' })), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
      ),
    )

    mountResultPanel('j1', 'clip.wav')

    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Open player' })).toBeInTheDocument(),
    )
  })

  it('hides the control when the upload is gone', async () => {
    // A deleted or restarted job has no bytes left; a control that only 404s
    // is worse than no control.
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify(makeSnapshot({ media_url: null })), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
      ),
    )

    mountResultPanel('j1', 'clip.wav')

    await waitFor(() => expect(screen.getByText('Hello world')).toBeInTheDocument())
    expect(screen.queryByRole('button', { name: 'Open player' })).not.toBeInTheDocument()
  })

  it('opens the player from the tab bar without seeking', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify(makeSnapshot({ media_url: '/api/v1/jobs/j1/media?token=t' })), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
      ),
    )

    mountResultPanel('j1', 'clip.wav')
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Open player' })).toBeInTheDocument(),
    )
    fireEvent.click(screen.getByRole('button', { name: 'Open player' }))

    const state = useMediaPlayerStore.getState()
    expect(state.session?.jobId).toBe('j1')
    expect(state.session?.fileName).toBe('clip.wav')
    expect(state.seekRequest).toBeNull()
  })
})
