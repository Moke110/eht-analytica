<template>
  <Teleport to="body">
    <div class="modal-backdrop" v-if="visible">
      <div class="modal-box">
        <h3>Analysis</h3>
        <p class="progress-msg">{{ state.message }}</p>
        <div class="progress-bar-track">
          <div class="progress-bar-fill" :style="{ width: state.percent + '%' }"></div>
        </div>
        <p class="progress-pct">{{ Math.round(state.percent) }}%</p>

        <div class="modal-actions">
          <button
            v-if="state.status === 'running'"
            class="btn-danger"
            @click="$emit('cancel')"
          >Cancel</button>
          <button v-if="state.status !== 'running' && !saving" @click="$emit('close')">Close</button>
        </div>

        <!-- Save section after completion -->
        <div v-if="state.status === 'completed' && !saving" class="save-section">
          <hr />
          <button class="btn-primary" @click="saveResults" style="width:100%;">Save Results</button>
        </div>

        <div v-if="state.status === 'failed'" class="error-msg">{{ state.errorMessage }}</div>
        <div v-if="saveMessage" class="save-msg">{{ saveMessage }}</div>
      </div>
    </div>
  </Teleport>
</template>

<script setup>
import { ref } from 'vue'
import { apiPost } from '../../composables/useApi.js'

const props = defineProps({
  visible: { type: Boolean, default: false },
  state: { type: Object, default: () => ({ percent: 0, message: '', status: 'running' }) },
})

const emit = defineEmits(['cancel', 'close', 'save'])

const saving = ref(false)
const saveMessage = ref('')

async function saveResults() {
  saving.value = true
  try {
    const folderResp = await apiPost('/api/system/native-file-dialog', {
      type: 'directory',
      title: 'Select output folder',
    })
    if (!folderResp.paths || !folderResp.paths[0]) {
      saving.value = false
      return
    }
    const outputFolder = folderResp.paths[0]

    await apiPost('/api/analyze/save', {
      fs_results: props.state.result?.fs_results || [],
      metrics_rows: props.state.result?.metrics_rows || [],
      output_folder: outputFolder,
    })
    saveMessage.value = `Saved to ${outputFolder}`
  } catch (e) {
    saveMessage.value = `Error: ${e.message}`
  } finally {
    saving.value = false
  }
}
</script>

<style scoped>
.progress-msg { font-size: 13px; color: var(--text-muted); margin-bottom: 4px; }
.progress-pct { font-size: 12px; color: var(--text-muted); text-align: right; }
.error-msg { font-size: 12px; color: var(--danger); margin-top: 8px; }
.save-msg { font-size: 12px; color: #059669; margin-top: 8px; }
.save-section { margin-top: 12px; }
</style>
