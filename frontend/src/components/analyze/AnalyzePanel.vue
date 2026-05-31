<template>
  <div style="display:flex;flex-direction:column;height:100%;padding:16px;gap:10px;">
    <div style="display:flex;gap:8px;align-items:center;">
      <FilePicker
        button-label="Add Folder"
        dialog-type="directory"
        title="Select folder with CSV files"
        @paths-selected="onAddFolder"
      />
      <FilePicker
        button-label="Add CSV"
        dialog-type="file"
        title="Select CSV files"
        :filters="[['CSV files', '*.csv']]"
        :multi="true"
        @paths-selected="onAddCsv"
      />
      <button class="btn-danger" @click="removeSelected" :disabled="!csvPaths.length">Delete Selected</button>
      <button @click="clearAll" :disabled="!csvPaths.length">Clear List</button>
      <span style="flex:1;"></span>
      <button class="btn-primary" @click="onAnalyze" :disabled="!csvPaths.length">Analyze</button>
    </div>

    <CsvList
      ref="csvListRef"
      :paths="csvPaths"
      @update:paths="csvPaths = $event"
    />

    <AnalyzeProgressDialog
      :visible="progressVisible"
      :state="progressState"
      @cancel="cancelAnalysis"
      @close="progressVisible = false"
    />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import FilePicker from '../shared/FilePicker.vue'
import CsvList from './CsvList.vue'
import AnalyzeProgressDialog from './AnalyzeProgressDialog.vue'
import { apiPost } from '../../composables/useApi.js'
import { useTaskProgress } from '../../composables/useTaskProgress.js'

const csvListRef = ref(null)
const csvPaths = ref([])

const { state: progressState, start: startProgress, cancel: cancelProgress } = useTaskProgress()
const progressVisible = ref(false)

async function onAddFolder(paths) {
  if (!paths[0]) return
  // List CSV files in folder via server
  try {
    const resp = await apiPost('/api/system/list-csv', { folder: paths[0] })
    if (resp.paths) {
      addPaths(resp.paths)
    }
  } catch {
    // fallback: use folder path - user can manually add CSVs
  }
}

async function onAddCsv(paths) {
  if (!paths.length) return
  const resp = await apiPost('/api/analyze/validate', { paths })
  addPaths(resp.valid)
}

function addPaths(paths) {
  for (const p of paths) {
    if (!csvPaths.value.includes(p)) {
      csvPaths.value.push(p)
    }
  }
}

function removeSelected() {
  csvListRef.value?.removeSelected()
}

function clearAll() {
  csvListRef.value?.clearAll()
}

async function onAnalyze() {
  if (!csvPaths.value.length) return
  progressVisible.value = true
  try {
    const resp = await apiPost('/api/analyze/start', { csv_paths: csvPaths.value })
    startProgress(resp.task_id, 'analyze')
  } catch (e) {
    progressState.status = 'failed'
    progressState.errorMessage = e.message
  }
}

function cancelAnalysis() {
  cancelProgress()
}
</script>
