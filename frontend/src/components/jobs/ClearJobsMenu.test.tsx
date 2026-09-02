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
    error_code: null,
    skipped: false,
    skip_reason_code: null,
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
    expect(screen.getByRole('button', { name: /Clear jobs/ })).toBeDisabled()
  })

  it('opens the menu with finished + all counts', () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('a', 'completed'), mkJob('b', 'running'), mkJob('c', 'failed')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear jobs/ }))
    expect(screen.getByRole('menuitem', { name: 'Clear finished (2)' })).toBeInTheDocument()
    expect(screen.getByRole('menuitem', { name: 'Clear all (3)' })).toBeInTheDocument()
  })

  it('disables "Clear finished" when only active jobs exist', () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('a', 'running'), mkJob('b', 'queued')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear jobs/ }))
    const finished = screen.getByRole('menuitem', { name: 'Clear finished (0)' })
    // `aria-disabled`, not the attribute: a disabled button shows no tooltip,
    // and the row still has to be able to say why it is unavailable.
    expect(finished).toHaveAttribute('aria-disabled', 'true')
    fireEvent.click(finished)
    expect(screen.queryByRole('button', { name: 'Clear' })).toBeNull()
    expect(screen.getByRole('menuitem', { name: 'Clear all (2)' })).not.toHaveAttribute(
      'aria-disabled',
    )
  })

  it('deletes every job on confirmed "Clear all"', async () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('a', 'completed'), mkJob('b', 'running')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear jobs/ }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Clear all (2)' }))
    expect(screen.getByText(/Remove 2 jobs\?/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
    await waitFor(() => expect(mockedDelete).toHaveBeenCalledTimes(2))
    expect(mockedDelete).toHaveBeenCalledWith('a')
    expect(mockedDelete).toHaveBeenCalledWith('b')
  })

  it('deletes only finished jobs on confirmed "Clear finished"', async () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('done', 'completed'), mkJob('run', 'running')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear jobs/ }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Clear finished (1)' }))
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
    await waitFor(() => expect(mockedDelete).toHaveBeenCalledTimes(1))
    expect(mockedDelete).toHaveBeenCalledWith('done')
    expect(mockedDelete).not.toHaveBeenCalledWith('run')
  })

  it('cancels without deleting and returns to the menu', () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('a', 'completed')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear jobs/ }))
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
    fireEvent.click(screen.getByRole('button', { name: /Clear jobs/ }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Clear all (2)' }))
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
    await waitFor(() => expect(screen.getByText('Cleared 1 of 2; 1 failed')).toBeInTheDocument())
  })

  it('closes on Escape, and only this layer', () => {
    const onAncestorKey = vi.fn()
    renderMenu(
      <div onKeyDown={onAncestorKey}>
        <ClearJobsMenu jobs={[mkJob('a', 'completed')]} />
      </div>,
    )
    const trigger = screen.getByRole('button', { name: /Clear jobs/ })
    fireEvent.click(trigger)
    // Opening moves focus onto the first item, so Escape is pressed there —
    // the menu catches it itself rather than listening on `document`, which is
    // what stops one press from also closing a dialog around it.
    fireEvent.keyDown(document.activeElement!, { key: 'Escape' })
    expect(screen.queryByRole('menu')).toBeNull()
    expect(trigger).toHaveFocus()
    expect(onAncestorKey).not.toHaveBeenCalled()
  })

  it('closes on an outside press', () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('a', 'completed')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear jobs/ }))
    expect(screen.getByRole('menu')).toBeInTheDocument()
    fireEvent.mouseDown(document.body)
    expect(screen.queryByRole('menu')).toBeNull()
  })

  it('forgets a pending confirmation when it closes', () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('a', 'completed')]} />)
    const trigger = screen.getByRole('button', { name: /Clear jobs/ })
    fireEvent.click(trigger)
    fireEvent.click(screen.getByRole('menuitem', { name: 'Clear all (1)' }))
    expect(screen.getByRole('button', { name: 'Clear' })).toBeInTheDocument()
    fireEvent.mouseDown(document.body)
    fireEvent.click(trigger)
    // Reopening on yesterday's question, one click from destroying the list,
    // is the failure mode this guards.
    expect(screen.getByRole('menuitem', { name: 'Clear all (1)' })).toBeInTheDocument()
  })

  it('invalidates the jobs query after a confirmed clear', async () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
    const invalidateSpy = vi.spyOn(qc, 'invalidateQueries')
    render(
      <QueryClientProvider client={qc}>
        <ClearJobsMenu jobs={[mkJob('a', 'completed')]} />
      </QueryClientProvider>,
    )
    fireEvent.click(screen.getByRole('button', { name: /Clear jobs/ }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Clear all (1)' }))
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
    await waitFor(() => expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['jobs'] }))
  })
})
