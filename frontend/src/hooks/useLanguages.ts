import { useQuery } from '@tanstack/react-query'
import { getLanguages } from '../api/meta'

export function useLanguages() {
  return useQuery({ queryKey: ['languages'], queryFn: getLanguages, staleTime: Infinity })
}
