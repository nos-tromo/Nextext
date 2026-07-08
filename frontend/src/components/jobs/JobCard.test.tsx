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
