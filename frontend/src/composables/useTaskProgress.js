import { reactive, ref } from 'vue'
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

  function start(tid, taskType) {
    taskId.value = tid
    state.status = 'running'
    state.percent = 0
    state.message = 'Starting...'
    state.errorMessage = ''
    state.result = null

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
      },
      onError(data) {
        state.status = 'failed'
        state.errorMessage = data.message
      },
      onCancelled() {
        state.status = 'cancelled'
        state.message = 'Cancelled'
      },
    })
  }

  async function cancel() {
    if (!taskId.value) return
    await apiDelete(`/api/track/task/${taskId.value}`)
    close()
    state.status = 'cancelled'
  }

  function reset() {
    close()
    taskId.value = ''
    state.status = 'idle'
    state.percent = 0
    state.message = ''
    state.result = null
  }

  return { state, start, cancel, reset }
}
