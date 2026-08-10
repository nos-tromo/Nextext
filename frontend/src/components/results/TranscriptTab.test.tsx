import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { TranscriptTab } from './TranscriptTab'
import type { TranscriptSegment } from '../../api/types'

const transcribeSegments: TranscriptSegment[] = [
  { start: '0.00', end: '2.00', speaker: null, text: 'Hello world', translation: null },
]

const translateSegments: TranscriptSegment[] = [
  { start: '0.00', end: '2.00', speaker: null, text: 'Hello world', translation: 'Hallo Welt' },
]

describe('TranscriptTab download buttons', () => {
  it('shows a single TXT button for a transcribe-only transcript', () => {
    render(<TranscriptTab jobId="j1" segments={transcribeSegments} stem="clip" />)
    expect(screen.getByRole('button', { name: 'Download TXT' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Download Transcript TXT' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Download Translation TXT' })).not.toBeInTheDocument()
  })

  it('splits into Transcript TXT and Translation TXT when a translation exists', () => {
    render(<TranscriptTab jobId="j1" segments={translateSegments} stem="clip" />)
    expect(screen.getByRole('button', { name: 'Download Transcript TXT' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Download Translation TXT' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Download TXT' })).not.toBeInTheDocument()
  })
})
