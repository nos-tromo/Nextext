import { useQuery } from '@tanstack/react-query'
import { getHealth } from '../api/meta'

export function useHealth() {
  return useQuery({ queryKey: ['health'], queryFn: getHealth })
}
