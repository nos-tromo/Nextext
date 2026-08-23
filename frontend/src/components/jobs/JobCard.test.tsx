import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'
import type { JobListItem, JobStatus } from '../../api/types'

vi.mock('../results/ResultPanel', () => ({ ResultPanel: () => null }))
vi.mock('../../api/jobs', () => ({
  deleteJob: vi.fn(),
  listJobs: vi.fn(),
  submitJob: vi.fn(),
}))

import { deleteJob } from '../../api/jobs'
import { JobCard } from './JobCard'
import { useJobProgressStore } from '../../lib/jobProgressStore'

const mockedDelete = vi.mocked(deleteJob)

function mkJob(job_id: string, status: JobStatus): JobListItem {
  return {
    job_id,
    status,
    file_name: `${job_id}.wav`,
    stage: null,
    progress: 0,
    error: null,
    created_at: 't',
    started_at: null,
    finished_at: null,
    task: 'transcribe',
    error_code: null,
    skipped: false,
    skip_reason_code: null,
  }
}

function renderCard(ui: ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

beforeEach(() => {
  mockedDelete.mockReset()
  mockedDelete.mockResolvedValue(undefined)
  useJobProgressStore.getState().clear()
})
afterEach(() => vi.restoreAllMocks())

describe('JobCard progress', () => {
  it('renders live per-job progress read from the shared store', () => {
    useJobProgressStore.getState().setJobProgress('j1', {
      status: 'running',
      stageIndex: 1,
      stageLabel: 'Translating',
      progress: 0.4,
      error: null,
      skipped: false,
      skipReason: null,
      errorCode: null,
      terminal: false,
    })
    renderCard(<JobCard job={mkJob('j1', 'running')} />)
    expect(screen.getByText(/Translating/)).toBeInTheDocument()
    expect(screen.getByText(/40%/)).toBeInTheDocument()
  })

  it('falls back to the list snapshot status when the stream has no entry yet', () => {
    renderCard(<JobCard job={mkJob('j2', 'completed')} />)
    expect(screen.getByText('Complete')).toBeInTheDocument()
  })
})

describe('JobCard failure message', () => {
  it('shows the interrupted-specific message for an interrupted job', () => {
    renderCard(<JobCard job={mkJob('j1', 'interrupted')} />)
    expect(
      screen.getByText('Job was interrupted before it could finish.'),
    ).toBeInTheDocument()
  })

  it('shows the generic unknown-error message for any other failed job', () => {
    renderCard(<JobCard job={mkJob('j1', 'failed')} />)
    expect(screen.getByText('Unknown error')).toBeInTheDocument()
  })
})

describe('JobCard Remove', () => {
  it('deletes the job when Remove is clicked', async () => {
    renderCard(<JobCard job={mkJob('j1', 'completed')} />)
    fireEvent.click(screen.getByRole('button', { name: 'Remove' }))
    await waitFor(() => expect(mockedDelete).toHaveBeenCalledWith('j1'))
  })

  it('shows an inline error when removal fails', async () => {
    mockedDelete.mockRejectedValue(new Error('nope'))
    renderCard(<JobCard job={mkJob('j1', 'completed')} />)
    fireEvent.click(screen.getByRole('button', { name: 'Remove' }))
    await waitFor(() => expect(screen.getByText(/Could not remove job/)).toBeInTheDocument())
  })
})

describe('JobCard skipped message', () => {
  it.each([
    ['vad_no_speech', 'Skipped — no speech detected in the audio'],
    ['asr_empty_transcript', 'Skipped — transcription returned no text'],
    ['asr_all_segments_filtered', 'Skipped — only non-speech audio was detected'],
  ] as const)('names the reason for %s', (code, text) => {
    useJobProgressStore.getState().setJobProgress('j1', {
      status: 'completed',
      stageIndex: 0,
      stageLabel: null,
      progress: 1,
      error: null,
      skipped: true,
      skipReason: code,
      errorCode: null,
      terminal: true,
    })
    renderCard(<JobCard job={{ ...mkJob('j1', 'completed'), skipped: true, skip_reason_code: code }} />)
    expect(screen.getByText(text)).toBeInTheDocument()
  })

  it('still reports the skip after a reload, with only the job list to go on', () => {
    // No live store entry — exactly the post-refresh state.
    renderCard(
      <JobCard job={{ ...mkJob('j1', 'completed'), skipped: true, skip_reason_code: 'vad_no_speech' }} />,
    )
    expect(screen.getByText('Skipped — no speech detected in the audio')).toBeInTheDocument()
  })

  it('keeps the warning icon sized (its default className is replaced, not merged)', () => {
    const { container } = renderCard(
      <JobCard job={{ ...mkJob('j1', 'completed'), skipped: true, skip_reason_code: 'vad_no_speech' }} />,
    )
    const icon = container.querySelector('svg')
    expect(icon).not.toBeNull()
    expect(icon?.getAttribute('class')).toMatch(/\bh-4\b/)
    expect(icon?.getAttribute('class')).toMatch(/\bw-4\b/)
  })

  it('explains an undecodable upload instead of calling it an unknown error', () => {
    renderCard(<JobCard job={{ ...mkJob('j1', 'failed'), error_code: 'undecodable_media' }} />)
    expect(screen.getByText(/could not be decoded/)).toBeInTheDocument()
  })
})
