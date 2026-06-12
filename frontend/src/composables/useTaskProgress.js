import { reactive, ref, onBeforeUnmount } from 'vue'
import { useSse } from './useSse.js'
import { apiDelete } from './useApi.js'

export function useTaskProgress() {
  const state = reactive({
    percent: 0,
    message: '',
    status: 'idle', // idle | running | completed | failed | cancelled
    errorMessage: '',
    result: null,
  })
  const taskId = ref('')
  const { connect, close } = useSse()

  let _taskType = 'track'

  function onPageUnload() {
    if (taskId.value && state.status === 'running') {
      // fetch with keepalive: true for reliable delivery during page unload
      fetch(`/api/${_taskType}/task/${taskId.value}`, { method: 'DELETE', keepalive: true })
    }
  }

  function start(tid, taskType, onComplete) {
    taskId.value = tid
    _taskType = taskType || 'track'
    state.status = 'running'
    state.percent = 0
    state.message = 'Starting...'
    state.errorMessage = ''
    state.result = null

    window.addEventListener('beforeunload', onPageUnload)

    const streamUrl = `/api/${taskType}/task/${tid}/stream`
    connect(streamUrl, {
      onProgress(data) {
        state.percent = data.percent
        state.message = data.message
      },
      onComplete(data) {
        state.status = 'completed'
        state.percent = 100
        state.result = data.result
        state.message = data.message
        window.removeEventListener('beforeunload', onPageUnload)
        if (onComplete) onComplete(data)
      },
      onError(data) {
        state.status = 'failed'
        state.errorMessage = data.message
        window.removeEventListener('beforeunload', onPageUnload)
      },
      onCancelled() {
        state.status = 'cancelled'
        state.message = 'Cancelled'
        window.removeEventListener('beforeunload', onPageUnload)
      },
      onDisconnect() {
        if (state.status === 'running') {
          cancel()
        }
        window.removeEventListener('beforeunload', onPageUnload)
      },
    })
  }

  async function cancel() {
    if (!taskId.value) return
    await apiDelete(`/api/${_taskType}/task/${taskId.value}`)
    close()
    state.status = 'cancelled'
    window.removeEventListener('beforeunload', onPageUnload)
  }

  function reset() {
    close()
    taskId.value = ''
    state.status = 'idle'
    state.percent = 0
    state.message = ''
    state.result = null
    window.removeEventListener('beforeunload', onPageUnload)
  }

  onBeforeUnmount(reset)

  return { state, start, cancel, reset }
}
