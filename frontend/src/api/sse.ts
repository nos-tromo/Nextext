import { API_BASE, OWNER_HEADER } from './client'
import { resolveOwnerId } from '../identity/owner'

export interface SseFrame {
  event: string
  data: string
}

/** Incremental SSE frame parser: feed raw text via push(), drain with flush(). */
export function parseSseFrames() {
  let buffer = ''

  function drain(): SseFrame[] {
    const frames: SseFrame[] = []
    let sep: number
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const raw = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      let event = ''
      const dataLines: string[] = []
      for (const line of raw.split('\n')) {
        if (line.startsWith(':')) continue
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
      }
      if (event && dataLines.length) frames.push({ event, data: dataLines.join('\n') })
    }
    return frames
  }

  return {
    push(chunk: string): SseFrame[] {
      buffer += chunk
      return drain()
    },
    flush(): SseFrame[] {
      if (buffer && !buffer.endsWith('\n\n')) buffer += '\n\n'
      return drain()
    },
  }
}

/**
 * Open a GET SSE stream at `${API_BASE}${path}` and yield parsed frames until
 * the stream closes or `signal` aborts. The backend replays history on connect,
 * so a fresh call after a drop resumes from the beginning (caller dedupes).
 */
export async function* streamSse(
  path: string,
  signal?: AbortSignal,
): AsyncGenerator<SseFrame> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { [OWNER_HEADER]: resolveOwnerId(), accept: 'text/event-stream' },
    signal,
  })
  if (!res.ok || !res.body) {
    throw new Error(`SSE ${res.status} for ${path}`)
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  const parser = parseSseFrames()
  try {
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      for (const frame of parser.push(decoder.decode(value, { stream: true }))) {
        yield frame
      }
    }
    for (const frame of parser.flush()) yield frame
  } finally {
    reader.releaseLock()
  }
}
