<template>
  <div class="panel-h">
    <TrackSidebar
      ref="sidebarRef"
      @video-opened="onVideoOpened"
      @model-loaded="onModelLoaded"
      @track-start="onTrackStart"
      @roi-names-loaded="onRoiNamesLoaded"
      @job-dir-created="onJobDirCreated"
    />
    <div class="canvas-area">
      <VideoCanvas
        ref="canvasRef"
        :frame-src="frameBase64"
        :frame-size="frameSize"
        :saved-names="savedNames"
        @rois-changed="onRoisChanged"
        @names-changed="onNamesChanged"
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

const emit = defineEmits(['job-dir-changed'])

const sidebarRef = ref(null)
const canvasRef = ref(null)

const frameBase64 = ref('')
const frameSize = reactive({ width: 0, height: 0 })
const metadata = ref(null)
const videoId = ref('')
const jobDir = ref('')

const { state: progressState, start: startProgress, cancel: cancelProgress } = useTaskProgress()
const progressVisible = ref(false)

const savedNames = ref([])

function onRoiNamesLoaded(names) {
  savedNames.value = names || []
}

function onNamesChanged(names) {
  savedNames.value = names
  apiPost('/api/system/config/save-roi-names', { names }).catch(() => {})
}

function onJobDirCreated(dir) {
  jobDir.value = dir
  emit('job-dir-changed', dir)
}

function onVideoOpened(data) {
  videoId.value = data.videoId
  metadata.value = data.metadata
  frameSize.width = data.metadata.width
  frameSize.height = data.metadata.height
  frameBase64.value = data.firstFrame
  if (canvasRef.value) canvasRef.value.clearRois()
}

function onModelLoaded() { }

function onRoisChanged(count) {
  if (sidebarRef.value) sidebarRef.value.setRoiCount(count)
}

async function onTrackStart(payload) {
  const rois = canvasRef.value ? canvasRef.value.getRois() : []
  if (!rois.length) return

  progressVisible.value = true

  try {
    const resp = await apiPost('/api/track/start', {
      video_id: videoId.value,
      rois,
      output_folder: jobDir.value,
      save_tracked_video: payload.saveTrackedVideo || false,
      save_inferences: payload.saveInferences || false,
    })
    startProgress(resp.task_id, 'track')
  } catch (e) {
    progressState.status = 'failed'
    progressState.errorMessage = e.message
  }
}

function cancelTracking() {
  cancelProgress()
  progressVisible.value = false
}
</script>
