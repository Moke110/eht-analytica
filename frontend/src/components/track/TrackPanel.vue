<template>
  <div class="panel-h">
    <TrackSidebar
      ref="sidebarRef"
      @video-opened="onVideoOpened"
      @video-processed="onVideoProcessed"
      @model-loaded="onModelLoaded"
      @track-start="onTrackStart"
    />
    <div class="canvas-area">
      <VideoCanvas
        ref="canvasRef"
        :frame-src="frameBase64"
        :frame-size="frameSize"
        @rois-changed="onRoisChanged"
      />
    </div>
    <TrackProgressDialog
      :visible="progressVisible"
      title="Tracking"
      :state="progressState"
      @cancel="cancelTracking"
      @close="progressVisible = false"
    />
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import TrackSidebar from './TrackSidebar.vue'
import VideoCanvas from './VideoCanvas.vue'
import TrackProgressDialog from './TrackProgressDialog.vue'
import { apiPost } from '../../composables/useApi.js'
import { useTaskProgress } from '../../composables/useTaskProgress.js'

const sidebarRef = ref(null)
const canvasRef = ref(null)

const frameBase64 = ref('')
const frameSize = reactive({ width: 0, height: 0 })
const metadata = ref(null)
const videoId = ref('')

const { state: progressState, start: startProgress, cancel: cancelProgress, reset: resetProgress } = useTaskProgress()
const progressVisible = ref(false)

function onVideoOpened(data) {
  videoId.value = data.videoId
  metadata.value = data.metadata
  frameSize.width = data.metadata.width
  frameSize.height = data.metadata.height
  frameBase64.value = data.firstFrame
}

function onVideoProcessed(result) {
  // Video pre-processing complete — frame buffer ready on server
  // Set output folder
  if (sidebarRef.value && metadata.value) {
    const videoDir = metadata.value.path ? metadata.value.path.substring(0, metadata.value.path.lastIndexOf('\\')) : ''
    const outputPath = videoDir ? `${videoDir}\\EHT_analytics` : ''
    sidebarRef.value.setOutputFolder(outputPath)
  }
}

function onModelLoaded() {
  // Model loaded on server
}

function onRoisChanged(count) {
  if (sidebarRef.value) sidebarRef.value.setRoiCount(count)
}

async function onTrackStart({ videoId: vid }) {
  const rois = canvasRef.value ? canvasRef.value.getRois() : []
  if (!rois.length) return

  const outputFolder = `${metadata.value.path.substring(0, metadata.value.path.lastIndexOf('\\'))}\\EHT_analytics`

  progressVisible.value = true

  try {
    const resp = await apiPost('/api/track/start', {
      video_id: vid,
      rois,
      output_folder: outputFolder,
    })
    startProgress(resp.task_id, 'track')
  } catch (e) {
    progressState.status = 'failed'
    progressState.errorMessage = e.message
  }
}

function cancelTracking() {
  cancelProgress()
}
</script>
