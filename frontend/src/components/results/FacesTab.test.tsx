import { afterEach, describe, expect, it } from 'vitest'
import { act, fireEvent, render, screen } from '@testing-library/react'
import { FacesTab } from './FacesTab'
import type { FaceEntry, FacesResult } from '../../api/types'
import { useMediaPlayerStore } from '../../lib/mediaPlayerStore'

// A placeholder image: one transparent pixel, no face.
const THUMB = 'data:image/gif;base64,R0lGODlhAQABAAAAACw='
const MEDIA_URL = '/api/v1/jobs/j1/media?token=invented'

function entry(overrides: Partial<FaceEntry> = {}): FaceEntry {
  return {
    number: 1,
    assigned: true,
    first_seen_seconds: 1,
    last_seen_seconds: 7,
    appearances: [
      { start_seconds: 1, end_seconds: 3 },
      { start_seconds: 6, end_seconds: 8 },
    ],
    detections: 3,
    thumbnail: THUMB,
    best_artifact: 'face_001.jpg',
    ...overrides,
  }
}

function faces(overrides: Partial<FacesResult> = {}): FacesResult {
  return {
    outcome: 'ok',
    grouped: true,
    truncated: false,
    entries: [
      entry(),
      entry({
        number: 2,
        assigned: false,
        first_seen_seconds: 12,
        last_seen_seconds: 12,
        appearances: [{ start_seconds: 12, end_seconds: 13 }],
        detections: 1,
        best_artifact: 'face_002.jpg',
      }),
    ],
    ...overrides,
  }
}

function mount(result: FacesResult, { mediaUrl = null, groupCapable = true }: { mediaUrl?: string | null; groupCapable?: boolean } = {}) {
  return render(
    <FacesTab jobId="j1" faces={result} stem="clip" fileName="clip.mp4" mediaUrl={mediaUrl} groupCapable={groupCapable} />,
  )
}

afterEach(() => useMediaPlayerStore.getState().clear())

describe('FacesTab', () => {
  it('renders one card per entry with thumbnail and localized label', () => {
    mount(faces())

    const person = screen.getByAltText('Person 1')
    expect(person).toHaveAttribute('src', THUMB)
    expect(screen.getByAltText('Face 2')).toBeInTheDocument()
    expect(screen.getAllByRole('listitem')).toHaveLength(2)
    expect(screen.getByText('3 sightings')).toBeInTheDocument()
    expect(screen.getByText('1 sighting')).toBeInTheDocument()
    expect(screen.getByText('First seen 00:01 · last seen 00:07')).toBeInTheDocument()
  })

  it('appearance button seeks the player', () => {
    mount(faces(), { mediaUrl: MEDIA_URL })

    fireEvent.click(screen.getByRole('button', { name: 'Play from 00:06' }))

    const state = useMediaPlayerStore.getState()
    expect(state.session?.jobId).toBe('j1')
    expect(state.seekRequest?.seconds).toBe(6)
  })

  it('shows appearances as plain text without media', () => {
    mount(faces())

    expect(screen.queryByRole('button', { name: /Play from/ })).not.toBeInTheDocument()
    expect(screen.getByText('00:06')).toBeInTheDocument()
  })

  it('highlights the entry on screen', () => {
    mount(faces(), { mediaUrl: MEDIA_URL })

    act(() => {
      useMediaPlayerStore.getState().open({ jobId: 'j1', fileName: 'clip.mp4', mediaUrl: MEDIA_URL })
      useMediaPlayerStore.getState().setCurrentTime(7)
    })

    expect(screen.getByAltText('Person 1').closest('li')).toHaveAttribute('aria-current', 'true')
    expect(screen.getByAltText('Face 2').closest('li')).not.toHaveAttribute('aria-current')
  })

  it('does not highlight while another job is playing', () => {
    mount(faces(), { mediaUrl: MEDIA_URL })

    act(() => {
      useMediaPlayerStore.getState().open({ jobId: 'other', fileName: 'other.mp4', mediaUrl: MEDIA_URL })
      useMediaPlayerStore.getState().setCurrentTime(7)
    })

    expect(screen.getByAltText('Person 1').closest('li')).not.toHaveAttribute('aria-current')
  })

  it('shows the partial, truncated and not-grouped banners', () => {
    mount(faces({ outcome: 'partial', truncated: true, grouped: false }))

    expect(screen.getByText('Face detection was not reachable for part of the video.')).toBeInTheDocument()
    expect(screen.getByText('Only the most prominent faces are shown.')).toBeInTheDocument()
    expect(screen.getByText('Faces were not grouped by person.')).toBeInTheDocument()
  })

  it('says nothing about grouping where the deployment does not group', () => {
    mount(faces({ grouped: false }), { groupCapable: false })

    expect(screen.queryByText('Faces were not grouped by person.')).not.toBeInTheDocument()
  })

  it("leaves the tab's downloads to the result header", () => {
    mount(faces())

    expect(screen.queryByRole('button', { name: 'Download all faces (ZIP)' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Download face table (CSV)' })).not.toBeInTheDocument()
  })

  it("offers each best image as an icon at its card's right edge, beside the name", () => {
    mount(faces())

    const best = screen.getAllByRole('button', { name: 'Download best image' })
    expect(best).toHaveLength(2)
    expect(best[0].textContent).toBe('')
    const name = screen.getByText('Person 1')
    expect(name.parentElement).toContainElement(best[0])
    expect(name).toHaveClass('flex-1')
    // Pulled into the name's line, so the 32 px button does not push the card's text down.
    expect(best[0].closest('.-my-1\\.5')).not.toBeNull()
  })

  it('shows faces_none for an empty ok result', () => {
    mount(faces({ entries: [] }))

    expect(screen.getByText('No faces found.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Download all faces (ZIP)' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Download face table (CSV)' })).not.toBeInTheDocument()
  })

  it('shows only the unavailable banner when detection never answered', () => {
    mount(faces({ outcome: 'unavailable', grouped: false, entries: [] }))

    expect(screen.getByText('Face detection was not reachable.')).toBeInTheDocument()
    expect(screen.queryByText('No faces found.')).not.toBeInTheDocument()
    expect(screen.queryByText('Faces were not grouped by person.')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Download all faces (ZIP)' })).not.toBeInTheDocument()
  })
})
