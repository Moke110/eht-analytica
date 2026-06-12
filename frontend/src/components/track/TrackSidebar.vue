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
    <div class="model-server-info" v-if="modelServerInfo">{{ modelServerInfo }}</div>
    <select
      class="model-dropdown"
      v-model="selectedModelName"
      @change="onModelDropdownChanged"
      :disabled="modelLoading"
    >
      <option value="" disabled>Select a model...</option>
      <option v-for="m in availableModels" :key="m.name" :value="m.name">
        {{ m.display_name }}
      </option>
    </select>
    <div v-if="modelLoading" class="loading-indicator">Loading model...</div>

    <hr />

    <!-- Track -->
    <label class="checkbox-label" style="margin-top:8px;">
      <input type="checkbox" v-model="saveTrackedVideo" />
      Save tracked video
    </label>
    <label class="checkbox-label">
      <input type="checkbox" v-model="saveInferences" />
      Save inferences
    </label>
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
  'model-loaded',
  'track-start',
  'track-complete',
  'roi-names-loaded',
  'job-dir-created',
])

const videoId = ref('')
const videoPath = ref('')
const preprocessState = ref('idle')
const preprocessStatus = ref('')
const modelDisplay = ref('')
const modelLoaded = ref(false)
const roiCount = ref(0)
const outputFolder = ref('')
const trackError = ref('')
const saveTrackedVideo = ref(false)
const saveInferences = ref(false)

// Model dropdown state
const availableModels = ref([])
const selectedModelName = ref('')
const modelLoading = ref(false)

// Device state
const deviceName = ref('Detecting...')
const deviceLoaded = ref(false)

// Model server state
const modelServerInfo = ref('')

const deviceClass = computed(() =>
  deviceName.value.toLowerCase().includes('cuda') || deviceName.value.toLowerCase().includes('nvidia')
    ? 'device-gpu'
    : 'device-cpu'
)

const deviceDisplay = computed(() => {
  if (deviceLoaded.value) {
    return `✔ ${deviceName.value}`
  }
  return deviceName.value
})

onMounted(async () => {
  try {
    const info = await apiGet('/api/system/device')
    deviceName.value = info.device_info || info.device
    deviceLoaded.value = true
  } catch {
    deviceName.value = 'Unknown'
  }

  // Fetch available models for dropdown
  try {
    const resp = await apiGet('/api/system/models')
    availableModels.value = resp.models || []
  } catch (e) {
    console.error('Failed to fetch models:', e)
  }

  // Load saved ROI names
  try {
    const config = await apiGet('/api/system/config')
    if (config.roi_names && config.roi_names.length > 0) {
      emit('roi-names-loaded', config.roi_names)
    }
  } catch {}

  // Sync with model auto-loaded by App.vue (may still be loading)
  pollModelStatus()
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
    emit('video-opened', {
      videoId: resp.video_id,
      metadata: resp.metadata,
      firstFrame: resp.first_frame_base64,
    })

    // Init job directory
    try {
      const jobResp = await apiPost('/api/system/init-job-dir', { video_path: path })
      outputFolder.value = jobResp.job_dir
      emit('job-dir-created', jobResp.job_dir)
    } catch {
      const videoDir = path.substring(0, path.lastIndexOf('\\'))
      outputFolder.value = videoDir + '\\EHT-analytics'
    }

    preprocessState.value = 'ready'
    const durSec = resp.metadata.duration || 0
    const durStr = durSec >= 60
      ? `${Math.floor(durSec / 60)}m ${Math.round(durSec % 60)}s`
      : `${Math.round(durSec)}s`
    preprocessStatus.value = `Ready (${durStr})`
  } catch (e) {
    preprocessState.value = 'error'
    preprocessStatus.value = e.message
    trackError.value = e.message
  }
}

function onModelDropdownChanged() {
  if (!selectedModelName.value) return
  modelLoading.value = true
  trackError.value = ''
  loadModelByName(selectedModelName.value)
}

async function loadModelByName(modelName) {
  try {
    const resp = await apiPost('/api/track/model/load', { model_name: modelName })
    const es = new EventSource(`/api/track/task/${resp.task_id}/stream`)

    es.onmessage = (e) => {
      if (!e.data) return
      try {
        const d = JSON.parse(e.data)
        if (d.event === 'progress') {
          // model loading progress
        } else if (d.event === 'complete') {
          modelLoaded.value = true
          modelDisplay.value = d.result?.model_name || modelName
          // Show model status (name + device) above the dropdown
          const mName = d.result?.model_name || ''
          const device = d.result?.device || ''
          if (mName) {
            modelServerInfo.value = `${mName}  |  ${device}  ✓`
          }
          modelLoading.value = false
          emit('model-loaded', d.result)
          es.close()
        } else if (d.event === 'error') {
          trackError.value = d.message
          modelLoading.value = false
          es.close()
        }
      } catch {}
    }
    es.onerror = () => { modelLoading.value = false; es.close() }
  } catch (e) {
    trackError.value = e.message
    modelLoading.value = false
  }
}

function onTrack() {
  emit('track-start', {
    videoId: videoId.value,
    outputFolder: outputFolder.value,
    saveTrackedVideo: saveTrackedVideo.value,
    saveInferences: saveInferences.value,
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
    modelDisplay.value = status.model_name || ''
    selectedModelName.value = status.model_name || ''
    if (status.device) {
      modelServerInfo.value = `${status.model_name || ''}  |  ${status.device}  ✓`
    }
  }
}

async function pollModelStatus() {
  for (let i = 0; i < 15; i++) {
    try {
      const status = await apiGet('/api/track/model/status')
      if (status.loaded) {
        syncModelStatus(status)
        return
      }
    } catch {}
    await new Promise(r => setTimeout(r, 2000))
  }
}

defineExpose({ setRoiCount, setOutputFolder, videoId, videoPath })
</script>
