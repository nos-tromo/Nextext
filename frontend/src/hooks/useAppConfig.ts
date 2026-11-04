import { useQuery } from '@tanstack/react-query'
import { getConfig } from '../api/meta'
import type { AppConfig } from '../api/types'

/**
 * The SPA's bootstrap config (`GET /config`): the UI language and the audit
 * actions that need a stated purpose. Fetched once per session and shared —
 * the language provider holds the first paint for it, and the header reads
 * the purpose requirement from the same cache entry.
 */
export function useAppConfig() {
  return useQuery({
    queryKey: ['app-config'],
    queryFn: getConfig,
    staleTime: Infinity,
    gcTime: Infinity,
    retry: 1,
  })
}

/**
 * Whether the backend refuses some requests that state no purpose.
 *
 * @param config - The bootstrap config, if loaded.
 * @returns `true` when `AUDIT_REQUIRE_PURPOSE` lists any action.
 */
export function purposeRequired(config: AppConfig | undefined): boolean {
  return (config?.require_purpose ?? []).length > 0
}
