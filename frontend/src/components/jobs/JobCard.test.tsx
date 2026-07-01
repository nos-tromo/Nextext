import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'
import type { JobListItem, JobStatus } from '../../api/types'

vi.mock('../../hooks/useJobStream', () => ({
  useJobStream: () => ({
    status: 'completed',
    stageIndex: 0,
    stageLabel: null,
    progress: 1,
    error: null,
    skipped: false,
    terminal: true,
  }),
}))
vi.mock('../results/ResultPanel', () => ({ ResultPanel: () => null }))
vi.mock('../../api/jobs', () => ({
  deleteJob: vi.fn(),
  listJobs: vi.fn(),
  submitJob: vi.fn(),
}))

import { deleteJob } from '../../api/jobs'
import { JobCard } from './JobCard'

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
})
afterEach(() => vi.restoreAllMocks())

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
