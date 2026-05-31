const BASE = import.meta.env.DEV ? '' : window.location.origin

async function request(method, path, body) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  }
  if (body !== undefined) {
    opts.body = JSON.stringify(body)
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

export function apiPost(path, body) {
  return request('POST', path, body)
}

export function apiDelete(path) {
  return request('DELETE', path)
}
