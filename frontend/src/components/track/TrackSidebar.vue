<template>
  <div class="sidebar">
    <!-- Device -->
    <label>DEVICE</label>
    <div class="device-badge" :class="deviceClass">
      <span class="device-dot"></span>
      {{ deviceDisplay }}
    </div>

    <hr />

    <!-- Recording -->
    <label>Recording</label>
    <div class="path-text">{{ videoPath || 'No recording selected' }}</div>
    <FilePicker
      button-label="Select Recording"
      :title="'Select Video Recording'"
      :filters="[['Video files', '*.mp4;*.avi;*.mov;*.mkv']]"
      picker-key="last_recording_dir"
      @paths-selected="onVideoSelected"
    />

    <StatusIndicator
      :status-text="preprocessStatus"
      :status="preprocessState"
    />

    <hr />

    <!-- Model -->
    <label>Track Model</label>
    <div class="path-text">{{ modelDisplay || 'No model loaded' }}</div>
    <FilePicker
      button-label="Select Model"
      :title="'Select TorchScript Model'"
      :filters="[['TorchScript model', '*.pt']]"
      picker-key="last_model_dir"
      @paths-selected="onModelSelected"
    />

    <hr />

    <!-- Track -->
    <button
      class="btn-primary"
      :disabled="!canTrack"
      @click="onTrack"
      style="width:100%; margin-top:8px;"
    >Track</button>

    <button
      :disabled="!outputFolder"
      @click="openOutputFolder"
      style="width:100%;"
    >Open Output Folder</button>

    <div v-if="trackError" class="error-msg">{{ trackError }}</div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import FilePicker from '../shared/FilePicker.vue'
import StatusIndicator from '../shared/StatusIndicator.vue'
import { apiGet, apiPost } from '../../composables/useApi.js'

const emit = defineEmits([
  'video-opened',
  'video-processed',
  'model-loaded',
  'track-start',
  'track-complete',
])

const videoId = ref('')
const videoPath = ref('')
const preprocessState = ref('idle')
const preprocessStatus = ref('')
const modelPath = ref('')
const modelDisplay = ref('')
const modelLoaded = ref(false)
const roiCount = ref(0)
const outputFolder = ref('')
const trackError = ref('')

// Device state
const deviceName = ref('Detecting...')
const deviceLoaded = ref(false)

const deviceClass = computed(() =>
  deviceName.value.toLowerCase().includes('cuda') || deviceName.value.toLowerCase().includes('nvidia')
    ? 'device-gpu'
    : 'device-cpu'
)

const deviceDisplay = computed(() => {
  if (modelLoaded.value && modelDisplay.value) {
    return `✔ ${deviceName.value}`
  }
  return deviceName.value
})

onMounted(async () => {
  try {
    const info = await apiGet('/api/system/device')
    deviceName.value = info.device_info || info.device
  } catch {
    deviceName.value = 'Unknown'
  }

  // Sync with model that may already be loaded or still loading (App.vue auto-load)
  try {
    const status = await apiGet('/api/track/model/status')
    syncModelStatus(status)
  } catch {}

  // Re-check after model has had time to finish loading
  setTimeout(async () => {
    if (!modelLoaded.value) {
      try {
        const status = await apiGet('/api/track/model/status')
        syncModelStatus(status)
      } catch {}
    }
  }, 8000)
})

const canTrack = computed(() =>
  preprocessState.value === 'ready' && modelLoaded.value && roiCount.value > 0
)

function onVideoSelected(paths) {
  videoPath.value = paths[0]
  openVideo(paths[0])
}

async function openVideo(path) {
  preprocessState.value = 'running'
  preprocessStatus.value = 'Opening video...'
  trackError.value = ''

  try {
    const resp = await apiPost('/api/track/video/open', { path })
    videoId.value = resp.video_id
    preprocessStatus.value = 'Video pre-processing'
    emit('video-opened', {
      videoId: resp.video_id,
      metadata: resp.metadata,
      firstFrame: resp.first_frame_base64,
    })

    // Start frame scanning
    const procResp = await apiPost('/api/track/video/process', { path })
    const es = new EventSource(`/api/track/task/${procResp.task_id}/stream`)

    es.onmessage = (e) => {
      if (!e.data) return
      try {
        const d = JSON.parse(e.data)
        if (d.event === 'progress') {
          preprocessStatus.value = d.message
        } else if (d.event === 'complete') {
          preprocessState.value = 'ready'
          preprocessStatus.value = `Ready (${d.result?.valid_frame_count || 0} frames)`
          emit('video-processed', d.result)
          es.close()
        } else if (d.event === 'error') {
          preprocessState.value = 'error'
          preprocessStatus.value = d.message
          trackError.value = d.message
          es.close()
        }
      } catch {}
    }
    es.onerror = () => { es.close() }

  } catch (e) {
    preprocessState.value = 'error'
    preprocessStatus.value = e.message
    trackError.value = e.message
  }
}

function onModelSelected(paths) {
  modelPath.value = paths[0]
  loadModel(paths[0])
}

async function loadModel(path) {
  try {
    const resp = await apiPost('/api/track/model/load', { path })
    const es = new EventSource(`/api/track/task/${resp.task_id}/stream`)

    es.onmessage = (e) => {
      if (!e.data) return
      try {
        const d = JSON.parse(e.data)
        if (d.event === 'progress') {
          // model loading progress
        } else if (d.event === 'complete') {
          modelLoaded.value = true
          modelPath.value = path
          modelDisplay.value = d.result?.model_path || path
          if (d.result?.device_info) {
            deviceName.value = d.result.device_info
          }
          emit('model-loaded', d.result)
          es.close()
        } else if (d.event === 'error') {
          trackError.value = d.message
          es.close()
        }
      } catch {}
    }
    es.onerror = () => { es.close() }
  } catch (e) {
    trackError.value = e.message
  }
}

function onTrack() {
  emit('track-start', {
    videoId: videoId.value,
    outputFolder: outputFolder.value,
  })
}

function openOutputFolder() {
  if (outputFolder.value) {
    apiPost('/api/system/open-folder', { path: outputFolder.value })
  }
}

function setRoiCount(n) { roiCount.value = n }
function setOutputFolder(path) { outputFolder.value = path }

function syncModelStatus(status) {
  if (status.loaded) {
    modelLoaded.value = true
    modelPath.value = status.model_path || ''
    modelDisplay.value = status.model_path || status.model_name || ''
    if (status.device_info) {
      deviceName.value = status.device_info
    }
  }
}

defineExpose({ setRoiCount, setOutputFolder, videoId, videoPath })
</script>
