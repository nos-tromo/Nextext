import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'

vi.mock('../../lib/download', () => ({
  downloadArtifact: vi.fn(),
}))

import { downloadArtifact } from '../../lib/download'
import { ResultDownloadMenu } from './ResultDownloadMenu'
import type { DownloadSpec } from './DownloadButtons'

const mockedDownload = vi.mocked(downloadArtifact)

const ITEMS: DownloadSpec[] = [
  { name: 'transcript.csv', label: 'CSV', fileName: 'clip_transcript.csv' },
  { name: 'transcript.xlsx', label: 'XLSX', fileName: 'clip_transcript.xlsx' },
]

describe('ResultDownloadMenu', () => {
  beforeEach(() => {
    mockedDownload.mockReset()
  })

  it('opens a menu with every format of the tab', () => {
    render(<ResultDownloadMenu jobId="j1" tabLabel="Transcript" items={ITEMS} />)
    fireEvent.click(screen.getByRole('button', { name: 'Download Transcript' }))
    expect(screen.getByRole('menuitem', { name: 'CSV' })).toBeInTheDocument()
    expect(screen.getByRole('menuitem', { name: 'XLSX' })).toBeInTheDocument()
  })

  it('downloads the chosen format', async () => {
    mockedDownload.mockResolvedValue(undefined)
    render(<ResultDownloadMenu jobId="j1" tabLabel="Transcript" items={ITEMS} />)
    fireEvent.click(screen.getByRole('button', { name: 'Download Transcript' }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'XLSX' }))
    await waitFor(() =>
      expect(mockedDownload).toHaveBeenCalledWith('j1', 'transcript.xlsx', 'clip_transcript.xlsx'),
    )
  })

  it('reports a failed download beside the trigger', async () => {
    mockedDownload.mockRejectedValue(new Error('boom'))
    const { container } = render(<ResultDownloadMenu jobId="j1" tabLabel="Transcript" items={ITEMS} />)
    fireEvent.click(screen.getByRole('button', { name: 'Download Transcript' }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'CSV' }))
    await waitFor(() => expect(container.querySelector('.text-danger')).not.toBeNull())
  })

  it('holds the trigger while a download runs', async () => {
    mockedDownload.mockReturnValue(new Promise(() => {}))
    render(<ResultDownloadMenu jobId="j1" tabLabel="Transcript" items={ITEMS} />)
    fireEvent.click(screen.getByRole('button', { name: 'Download Transcript' }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'CSV' }))
    await waitFor(() => expect(screen.getByRole('button', { name: 'Download Transcript' })).toBeDisabled())
  })

  it('renders nothing for a tab without downloads', () => {
    const { container } = render(<ResultDownloadMenu jobId="j1" tabLabel="Transcript" items={[]} />)
    expect(container.firstChild).toBeNull()
  })
})
