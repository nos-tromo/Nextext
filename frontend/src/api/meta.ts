import { apiGet } from './client'
import type { HealthResponse, LanguagesResponse } from './types'

export function getHealth(): Promise<HealthResponse> {
  return apiGet<HealthResponse>('/health')
}

export function getLanguages(): Promise<LanguagesResponse> {
  return apiGet<LanguagesResponse>('/languages')
}

export function getVersion(): Promise<{ version: string }> {
  return apiGet<{ version: string }>('/version') // → GET /api/v1/version
}
