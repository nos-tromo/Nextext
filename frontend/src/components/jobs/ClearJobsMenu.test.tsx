import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'

vi.mock('../../api/jobs', () => ({
  deleteJob: vi.fn(),
  listJobs: vi.fn(),
  submitJob: vi.fn(),
}))

import { deleteJob } from '../../api/jobs'
import { ApiError } from '../../api/client'
import { ClearJobsMenu } from './ClearJobsMenu'
import type { JobListItem, JobStatus } from '../../api/types'

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

function renderMenu(ui: ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

beforeEach(() => {
  mockedDelete.mockReset()
  mockedDelete.mockResolvedValue(undefined)
})
afterEach(() => vi.restoreAllMocks())

describe('ClearJobsMenu', () => {
  it('disables the trigger when there are no jobs', () => {
    renderMenu(<ClearJobsMenu jobs={[]} />)
    expect(screen.getByRole('button', { name: /Clear/ })).toBeDisabled()
  })

  it('opens the menu with finished + all counts', () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('a', 'completed'), mkJob('b', 'running'), mkJob('c', 'failed')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear ▾/ }))
    expect(screen.getByRole('menuitem', { name: 'Clear finished (2)' })).toBeInTheDocument()
    expect(screen.getByRole('menuitem', { name: 'Clear all (3)' })).toBeInTheDocument()
  })

  it('disables "Clear finished" when only active jobs exist', () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('a', 'running'), mkJob('b', 'queued')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear ▾/ }))
    expect(screen.getByRole('menuitem', { name: 'Clear finished (0)' })).toBeDisabled()
    expect(screen.getByRole('menuitem', { name: 'Clear all (2)' })).toBeEnabled()
  })

  it('deletes every job on confirmed "Clear all"', async () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('a', 'completed'), mkJob('b', 'running')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear ▾/ }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Clear all (2)' }))
    expect(screen.getByText(/Remove 2 jobs\?/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
    await waitFor(() => expect(mockedDelete).toHaveBeenCalledTimes(2))
    expect(mockedDelete).toHaveBeenCalledWith('a')
    expect(mockedDelete).toHaveBeenCalledWith('b')
  })

  it('deletes only finished jobs on confirmed "Clear finished"', async () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('done', 'completed'), mkJob('run', 'running')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear ▾/ }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Clear finished (1)' }))
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
    await waitFor(() => expect(mockedDelete).toHaveBeenCalledTimes(1))
    expect(mockedDelete).toHaveBeenCalledWith('done')
    expect(mockedDelete).not.toHaveBeenCalledWith('run')
  })

  it('cancels without deleting and returns to the menu', () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('a', 'completed')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear ▾/ }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Clear all (1)' }))
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(mockedDelete).not.toHaveBeenCalled()
    expect(screen.getByRole('menuitem', { name: 'Clear all (1)' })).toBeInTheDocument()
  })

  it('reports a partial failure inline', async () => {
    mockedDelete.mockImplementation(async (id: string) => {
      if (id === 'bad') throw new ApiError(500, 'boom')
      return undefined
    })
    renderMenu(<ClearJobsMenu jobs={[mkJob('ok', 'completed'), mkJob('bad', 'failed')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear ▾/ }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Clear all (2)' }))
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
    await waitFor(() => expect(screen.getByText('Cleared 1 of 2; 1 failed')).toBeInTheDocument())
  })

  it('closes on Escape', () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('a', 'completed')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear ▾/ }))
    expect(screen.getByRole('menu')).toBeInTheDocument()
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(screen.queryByRole('menu')).toBeNull()
  })

  it('closes on outside pointerdown', () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('a', 'completed')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear ▾/ }))
    expect(screen.getByRole('menu')).toBeInTheDocument()
    fireEvent.pointerDown(document.body)
    expect(screen.queryByRole('menu')).toBeNull()
  })

  it('invalidates the jobs query after a confirmed clear', async () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
    const invalidateSpy = vi.spyOn(qc, 'invalidateQueries')
    render(
      <QueryClientProvider client={qc}>
        <ClearJobsMenu jobs={[mkJob('a', 'completed')]} />
      </QueryClientProvider>,
    )
    fireEvent.click(screen.getByRole('button', { name: /Clear ▾/ }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Clear all (1)' }))
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
    await waitFor(() => expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['jobs'] }))
  })
})
