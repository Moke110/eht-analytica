import { ref, onUnmounted } from 'vue'

export function useSse() {
  const eventSource = ref(null)
  const connected = ref(false)

  function connect(url, callbacks) {
    close()
    const es = new EventSource(url)
    eventSource.value = es
    connected.value = true

    es.onmessage = (e) => {
      if (!e.data) return
      try {
        const data = JSON.parse(e.data)
        if (data.event === 'progress' && callbacks.onProgress) {
          callbacks.onProgress(data)
        } else if (data.event === 'complete' && callbacks.onComplete) {
          callbacks.onComplete(data)
        } else if (data.event === 'error' && callbacks.onError) {
          callbacks.onError(data)
        } else if (data.event === 'cancelled' && callbacks.onCancelled) {
          callbacks.onCancelled(data)
        }
      } catch {}
    }

    es.onerror = () => {
      connected.value = false
      if (callbacks.onDisconnect) callbacks.onDisconnect()
    }
  }

  function close() {
    if (eventSource.value) {
      eventSource.value.close()
      eventSource.value = null
    }
    connected.value = false
  }

  onUnmounted(close)

  return { connect, close, connected }
}
