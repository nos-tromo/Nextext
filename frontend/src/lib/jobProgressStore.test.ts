import { beforeEach, describe, expect, it } from 'vitest'
import { initialJobProgress } from './jobProgress'
import { selectById, useJobProgressStore } from './jobProgressStore'

beforeEach(() => useJobProgressStore.getState().clear())

describe('jobProgressStore', () => {
  it('starts empty', () => {
    expect(useJobProgressStore.getState().byId).toEqual({})
  })

  it('setJobProgress inserts an entry', () => {
    const p = initialJobProgress('running')
    useJobProgressStore.getState().setJobProgress('j1', p)
    expect(useJobProgressStore.getState().byId.j1).toEqual(p)
  })

  it('setJobProgress replaces an existing entry', () => {
    useJobProgressStore.getState().setJobProgress('j1', initialJobProgress('queued'))
    useJobProgressStore.getState().setJobProgress('j1', initialJobProgress('completed'))
    expect(useJobProgressStore.getState().byId.j1.status).toBe('completed')
  })

  it('setJobProgress yields a new byId reference (the re-render trigger)', () => {
    const before = useJobProgressStore.getState().byId
    useJobProgressStore.getState().setJobProgress('j1', initialJobProgress('running'))
    expect(useJobProgressStore.getState().byId).not.toBe(before)
  })

  it('keeps multiple jobs independently', () => {
    useJobProgressStore.getState().setJobProgress('j1', initialJobProgress('running'))
    useJobProgressStore.getState().setJobProgress('j2', initialJobProgress('queued'))
    expect(Object.keys(useJobProgressStore.getState().byId).sort()).toEqual(['j1', 'j2'])
  })

  it('removeJob deletes an entry', () => {
    useJobProgressStore.getState().setJobProgress('j1', initialJobProgress('running'))
    useJobProgressStore.getState().removeJob('j1')
    expect(useJobProgressStore.getState().byId.j1).toBeUndefined()
  })

  it('removeJob on an absent key is a same-reference no-op', () => {
    useJobProgressStore.getState().setJobProgress('j1', initialJobProgress('running'))
    const before = useJobProgressStore.getState().byId
    useJobProgressStore.getState().removeJob('missing')
    expect(useJobProgressStore.getState().byId).toBe(before)
  })

  it('clear empties the map', () => {
    useJobProgressStore.getState().setJobProgress('j1', initialJobProgress('running'))
    useJobProgressStore.getState().clear()
    expect(useJobProgressStore.getState().byId).toEqual({})
  })

  it('selectById returns the byId map', () => {
    const state = useJobProgressStore.getState()
    expect(selectById(state)).toBe(state.byId)
  })
})
