const BASE = import.meta.env.DEV ? '' : window.location.origin

async function request(method, path, body, signal) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  }
  if (body !== undefined) {
    opts.body = JSON.stringify(body)
  }
  if (signal) {
    opts.signal = signal
  }
  const resp = await fetch(`${BASE}${path}`, opts)
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({ detail: resp.statusText }))
    throw new Error(detail.detail || resp.statusText)
  }
  return resp.json()
}

export function apiGet(path) {
  return request('GET', path)
}

export function apiPost(path, body, signal) {
  return request('POST', path, body, signal)
}

export function apiDelete(path) {
  return request('DELETE', path)
}
