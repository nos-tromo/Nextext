import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'

vi.mock('../../lib/download', () => ({
  downloadBatchArtifact: vi.fn(),
}))

import { downloadBatchArtifact } from '../../lib/download'
import { BatchDownloadMenu } from './BatchDownloadMenu'

const mockedDownload = vi.mocked(downloadBatchArtifact)

describe('BatchDownloadMenu', () => {
  beforeEach(() => {
    mockedDownload.mockReset()
  })
  afterEach(() => vi.restoreAllMocks())

  it('disables the trigger when no jobs have completed', () => {
    render(<BatchDownloadMenu completedCount={0} />)
    expect(screen.getByRole('button', { name: /Download all jobs/ })).toBeDisabled()
  })

  it('opens the menu with both options when jobs are available', () => {
    render(<BatchDownloadMenu completedCount={2} />)
    fireEvent.click(screen.getByRole('button', { name: /Download all jobs/ }))
    expect(screen.getByRole('menuitem', { name: 'Combined JSONL (docint)' })).toBeInTheDocument()
    expect(screen.getByRole('menuitem', { name: 'Full batch (ZIP)' })).toBeInTheDocument()
  })

  it('downloads the combined JSONL when that option is chosen', async () => {
    render(<BatchDownloadMenu completedCount={2} />)
    fireEvent.click(screen.getByRole('button', { name: /Download all jobs/ }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Combined JSONL (docint)' }))
    await waitFor(() => expect(mockedDownload).toHaveBeenCalledWith('docint.jsonl', 'nextext_docint.jsonl'))
  })

  it('downloads the full ZIP batch when that option is chosen', async () => {
    render(<BatchDownloadMenu completedCount={2} />)
    fireEvent.click(screen.getByRole('button', { name: /Download all jobs/ }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Full batch (ZIP)' }))
    await waitFor(() => expect(mockedDownload).toHaveBeenCalledWith('archive.zip', 'nextext_batch.zip'))
  })

  it('surfaces an inline error when the download fails', async () => {
    mockedDownload.mockRejectedValueOnce(new Error('no jobs to download'))
    render(<BatchDownloadMenu completedCount={1} />)
    fireEvent.click(screen.getByRole('button', { name: /Download all jobs/ }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Combined JSONL (docint)' }))
    await waitFor(() => expect(screen.getByText('no jobs to download')).toBeInTheDocument())
  })
})
