import { useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { PURPOSE_REQUIRED_STATUS, subscribePurpose } from '@infra/ui'
import { ApiError, PURPOSE_SCOPE } from '../api/client'

/**
 * Re-run the queries a missing purpose refused, once a purpose is stated.
 *
 * A request refused with `428` whose prompt the user dismissed leaves its
 * query failed — the jobs list, say, on first load. Stating a purpose in the
 * header bar afterwards refetches exactly those queries, so the page recovers
 * without a reload. Mounted once, in the Shell.
 */
export function useRetryOnPurpose(): void {
  const queryClient = useQueryClient()
  useEffect(
    () =>
      subscribePurpose(PURPOSE_SCOPE, (purpose) => {
        if (!purpose) return
        void queryClient.invalidateQueries({
          predicate: (query) =>
            query.state.error instanceof ApiError && query.state.error.status === PURPOSE_REQUIRED_STATUS,
        })
      }),
    [queryClient],
  )
}
