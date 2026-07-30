import { apiGet } from './client'
import type { AppConfig, HealthResponse, LanguagesResponse, WhoamiResponse } from './types'

export function getHealth(): Promise<HealthResponse> {
  return apiGet<HealthResponse>('/health')
}

export function getLanguages(): Promise<LanguagesResponse> {
  return apiGet<LanguagesResponse>('/languages')
}

export function getVersion(): Promise<{ version: string }> {
  return apiGet<{ version: string }>('/version') // → GET /api/v1/version
}

export function getConfig(): Promise<AppConfig> {
  return apiGet<AppConfig>('/config') // → GET /api/v1/config
}

/** Signed-in principal served by the backend's authenticated `/whoami` route. */
export function getWhoami(): Promise<WhoamiResponse> {
  return apiGet<WhoamiResponse>('/whoami') // → GET /api/v1/whoami
}
