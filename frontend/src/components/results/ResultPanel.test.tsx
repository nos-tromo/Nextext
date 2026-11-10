import { afterEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { JobSnapshot } from '../../api/types'
import { ResultPanel } from './ResultPanel'
import { useMediaPlayerStore } from '../../lib/mediaPlayerStore'
import { downloadArtifact } from '../../lib/download'

vi.mock('../../lib/download', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../lib/download')>()),
  downloadArtifact: vi.fn(async () => {}),
}))

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
      faces: false,
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
      faces: null,
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

  it('shows a silent clip\u2019s frame findings under the banner', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(
          JSON.stringify(
            makeSnapshot({
              transcript: [],
              skipped: true,
              skip_reason_code: 'vad_no_speech',
              frame_captions: [{ time_sec: 3, caption: 'A flag bearing a hate symbol.' }],
              hate_speech_findings: [
                {
                  hate_speech: true,
                  category: 'extremism',
                  confidence: 'high',
                  reason: 'Shows a hate symbol without distance.',
                  text: 'A flag bearing a hate symbol.',
                  start: '0:00:03',
                  speaker: null,
                  translation: null,
                  source: 'frame',
                },
              ],
            }),
          ),
          { status: 200, headers: { 'content-type': 'application/json' } },
        ),
      ),
    )

    mountResultPanel('j1', 'clip.mp4')

    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument())
    expect(screen.getByRole('button', { name: 'Hate Speech' })).toBeInTheDocument()
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

const FACES = {
  outcome: 'ok' as const,
  grouped: true,
  truncated: false,
  entries: [
    {
      number: 1,
      assigned: true,
      first_seen_seconds: 0,
      last_seen_seconds: 2,
      appearances: [{ start_seconds: 0, end_seconds: 3 }],
      detections: 3,
      // A placeholder image: one transparent pixel, no face.
      thumbnail: 'data:image/gif;base64,R0lGODlhAQABAAAAACw=',
      best_artifact: 'face_001.jpg',
    },
  ],
}

function serve(snapshot: JobSnapshot) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () =>
      new Response(JSON.stringify(snapshot), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    ),
  )
}

/** The header's right-hand cluster, found from the archive download it always holds. */
function cluster(): HTMLElement {
  return screen.getByRole('button', { name: 'Download all (.zip)' }).closest('.ml-auto') as HTMLElement
}

describe('ResultPanel downloads in the header', () => {
  it("offers the active tab's download menu in the header, before play and the archive", async () => {
    serve(makeSnapshot({ media_url: '/api/v1/jobs/j1/media?token=t' }))
    mountResultPanel('j1', 'clip.wav')
    await screen.findByText('Hello world')

    const names = Array.from(cluster().querySelectorAll('button')).map((b) => b.getAttribute('aria-label'))
    expect(names).toEqual(['Download Transcript', 'Open player', 'Download all (.zip)'])
    fireEvent.click(screen.getByRole('button', { name: 'Download Transcript' }))
    expect(screen.getAllByRole('menuitem').map((m) => m.textContent)).toEqual(['CSV', 'XLSX', 'TXT', 'JSONL'])
  })

  it('swaps the menu when the tab changes', async () => {
    serve(makeSnapshot({ summary: 'Ein kurzer Text.' }))
    mountResultPanel('j1', 'clip.wav')
    await screen.findByText('Hello world')

    fireEvent.click(screen.getByRole('button', { name: 'Summary' }))
    expect(within(cluster()).queryByRole('button', { name: 'Download Transcript' })).toBeNull()
    fireEvent.click(within(cluster()).getByRole('button', { name: 'Download Summary' }))
    expect(screen.getAllByRole('menuitem').map((m) => m.textContent)).toEqual(['TXT'])
  })

  it("starts the next tab's downloads clean", async () => {
    vi.mocked(downloadArtifact).mockRejectedValueOnce(new Error('boom'))
    serve(makeSnapshot({ summary: 'Ein kurzer Text.' }))
    mountResultPanel('j1', 'clip.wav')
    await screen.findByText('Hello world')

    fireEvent.click(within(cluster()).getByRole('button', { name: 'Download Transcript' }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'CSV' }))
    await waitFor(() => expect(cluster().querySelector('.text-danger')).not.toBeNull())
    fireEvent.click(screen.getByRole('button', { name: 'Summary' }))
    expect(cluster().querySelector('.text-danger')).toBeNull()
  })

  it('keeps the cluster right-aligned when the header wraps', async () => {
    serve(makeSnapshot())
    mountResultPanel('j1', 'clip.wav')
    await screen.findByText('Hello world')

    expect(cluster()).toHaveClass('ml-auto')
    expect(cluster().parentElement).toHaveClass('flex-wrap')
  })

  it('offers the keyframes ZIP for frames without captions', async () => {
    serve(makeSnapshot({ transcript: [], keyframes_url: '/api/v1/jobs/j1/artifacts/keyframes.zip' }))
    mountResultPanel('j1', 'clip.wav')

    expect(await screen.findByText(/sampled but not described/)).toBeInTheDocument()
    fireEvent.click(within(cluster()).getByRole('button', { name: 'Download Visual context' }))
    expect(screen.getByRole('menuitem', { name: 'Keyframes (ZIP)' })).toBeInTheDocument()
  })

  it('offers no tab menu for an empty transcript', async () => {
    serve(makeSnapshot({ transcript: [] }))
    mountResultPanel('j1', 'clip.wav')

    await screen.findByRole('button', { name: 'Download all (.zip)' })
    expect(within(cluster()).queryByRole('button', { name: /^Download Transcript$/ })).toBeNull()
  })

  it('shows an empty state when there is no result', async () => {
    serve({ ...makeSnapshot(), result: null } as unknown as JobSnapshot)
    mountResultPanel('j1', 'clip.wav')

    expect((await screen.findByText('No result data available.')).closest('.border-dashed')).not.toBeNull()
  })

  it('says which tab is active, since the header downloads belong to it', async () => {
    serve(makeSnapshot({ summary: 'Ein kurzer Text.' }))
    mountResultPanel('j1', 'clip.wav')
    await screen.findByText('Hello world')
    expect(screen.getByRole('button', { name: 'Transcript' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'Summary' })).toHaveAttribute('aria-pressed', 'false')
  })
})

describe('ResultPanel faces', () => {
  it('adds the faces tab after visual context', async () => {
    serve(makeSnapshot({
      summary: 'A brief summary.',
      frame_captions: [{ time_sec: 0, caption: 'A slide titled Roadmap' }],
      faces: FACES,
    }))

    mountResultPanel('j1', 'clip.mp4')

    await waitFor(() => expect(screen.getByRole('button', { name: 'Faces' })).toBeInTheDocument())
    const tabs = screen.getByRole('navigation').querySelectorAll('button')
    expect([...tabs].map((b) => b.textContent)).toEqual(['Transcript', 'Visual context', 'Faces', 'Summary'])
    fireEvent.click(screen.getByRole('button', { name: 'Faces' }))
    expect(screen.getByAltText('Person 1')).toBeInTheDocument()
  })

  it('keeps faces under the skipped banner', async () => {
    serve(makeSnapshot({ transcript: [], skipped: true, skip_reason_code: 'vad_no_speech', faces: FACES }))

    mountResultPanel('j1', 'clip.mp4')

    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument())
    expect(screen.getByRole('button', { name: 'Faces' })).toBeInTheDocument()
  })

  it('offers the faces downloads in the header menu', async () => {
    serve(makeSnapshot({ faces: FACES }))
    mountResultPanel('j1', 'clip.mp4')

    fireEvent.click(await screen.findByRole('button', { name: 'Faces' }))
    fireEvent.click(within(cluster()).getByRole('button', { name: 'Download Faces' }))
    expect(screen.getAllByRole('menuitem').map((m) => m.textContent)).toEqual(['Faces (ZIP)', 'Detections (CSV)'])
  })

  it('offers no faces menu when no face was found', async () => {
    serve(makeSnapshot({ faces: { ...FACES, entries: [] } }))
    mountResultPanel('j1', 'clip.mp4')

    fireEvent.click(await screen.findByRole('button', { name: 'Faces' }))
    expect(screen.getByText('No faces found.')).toBeInTheDocument()
    expect(within(cluster()).queryByRole('button', { name: 'Download Faces' })).toBeNull()
  })

  it('shows no faces tab when the job has no face result', async () => {
    serve(makeSnapshot({ faces: null }))

    mountResultPanel('j1', 'clip.mp4')

    await waitFor(() => expect(screen.getByRole('button', { name: 'Transcript' })).toBeInTheDocument())
    expect(screen.queryByRole('button', { name: 'Faces' })).not.toBeInTheDocument()
  })
})
