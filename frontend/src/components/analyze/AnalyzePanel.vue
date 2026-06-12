<template>
  <div class="analyze-root">
    <!-- Toolbar -->
    <div class="toolbar">
      <FilePicker
        button-label="Open"
        dialog-type="directory"
        title="Select EHT-analytics folder"
        @paths-selected="onOpenJobDir"
      />
      <div class="job-dir-path" :title="jobDir || ''">
        {{ jobDir || 'Open an EHT-analytics folder to begin' }}
      </div>
      <button
        class="btn-refresh"
        :disabled="!jobDir"
        @click="loadConfig"
        title="Refresh table from config.json"
      >Refresh</button>
      <span class="selected-count" v-if="selectedIds.size">
        {{ selectedIds.size }} selected
      </span>
      <button class="btn-primary" @click="onBatchAnalyze" :disabled="!canAnalyze">
        Analyze Selected
      </button>
    </div>

    <!-- Error -->
    <div v-if="errorMsg" class="error-banner">
      {{ errorMsg }}
      <button @click="errorMsg = ''">&times;</button>
    </div>

    <!-- Sample ID Table -->
    <div class="table-section">
      <div class="sample-table-wrapper">
        <table class="sample-table" v-if="samples.length">
          <thead>
            <tr>
              <th class="col-check">
                <input type="checkbox" :checked="allSelected" @change="toggleAll" />
              </th>
              <th>Sample ID</th>
              <th>Recording</th>
              <th>ROI Name</th>
              <th v-for="mk in metadataKeys" :key="mk" class="col-meta">
                <template v-if="editingMetaKey === mk">
                  <input v-model="editMetaKey" class="meta-key-input" />
                </template>
                <template v-else>{{ mk }}</template>
                <button v-if="editingMetaKey !== mk && !addingMetadata && !isMetaKeyBuiltIn(mk)" class="btn-icon meta-action" @click="startEditMeta(mk)" title="Edit">&#x270F;&#xFE0F;</button>
                <button v-if="editingMetaKey !== mk && !addingMetadata && !isMetaKeyBuiltIn(mk)" class="btn-icon meta-action" @click="deleteMeta(mk)" title="Delete">&#x1F5D1;&#xFE0F;</button>
                <button v-if="editingMetaKey === mk" class="btn-icon meta-action" @click="confirmEditMeta" title="Confirm">&#x2705;</button>
                <button v-if="editingMetaKey === mk" class="btn-icon meta-action" @click="cancelEditMeta" title="Cancel">&#x274C;</button>
              </th>
              <th class="col-meta-add">
                <button @click="startAddMetadata" :disabled="!jobDir || addingMetadata">Add Metadata</button>
                <div v-if="addingMetadata" class="meta-header-edit">
                  <input v-model="newMetaKey" placeholder="Key" class="meta-key-input" />
                  <button class="btn-confirm btn-icon" :disabled="!canConfirmMeta" @click="confirmMetadata">&#x2705;</button>
                  <button class="btn-cancel btn-icon" @click="cancelMetadata">&#x274C;</button>
                </div>
              </th>
              <th class="col-file">Length</th>
              <th class="col-file">Force&amp;Status</th>
              <th class="col-file">Metrics</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="acc in samples"
              :key="acc.id"
              :class="{ 'row-selected': selectedIds.has(acc.id) }"
            >
              <td class="col-check">
                <input
                  type="checkbox"
                  :checked="selectedIds.has(acc.id)"
                  @change="toggleOne(acc.id)"
                />
              </td>
              <td class="col-id">{{ acc.id }}</td>
              <td>{{ acc.recording_name }}</td>
              <td>{{ acc.roi_name }}</td>
              <td v-for="mk in metadataKeys" :key="mk" class="col-meta">
                <template v-if="editingMetaKey === mk">
                  <input v-model="editMetaValues[acc.id]" class="meta-val-input"
                         :data-acc="acc.id" :data-meta="mk"
                         @keydown.enter="focusNextMetaInput($event, acc.id, mk)" />
                </template>
                <template v-else>{{ acc.metadata?.[mk] || '' }}</template>
              </td>
              <td class="col-meta-editing">
                <input v-if="addingMetadata" v-model="newMetaValues[acc.id]" placeholder="value" class="meta-val-input" />
              </td>
              <td class="col-file">
                <span
                  v-if="acc.length_csv"
                  class="check-mark"
                  :title="acc.length_csv"
                  @click="openFile(acc.length_csv)"
                >&#x2705;</span>
                <span v-else class="cross-mark" title="Not available">&#x274C;</span>
              </td>
              <td class="col-file">
                <span
                  v-if="acc.force_status_csv"
                  class="check-mark"
                  :title="acc.force_status_csv"
                  @click="openFile(acc.force_status_csv)"
                >&#x2705;</span>
                <span v-else class="cross-mark" title="Not available">&#x274C;</span>
              </td>
              <td class="col-file">
                <span
                  v-if="acc.metrics_csv"
                  class="check-mark"
                  :title="acc.metrics_csv"
                  @click="openFile(acc.metrics_csv)"
                >&#x2705;</span>
                <span v-else class="cross-mark" title="Not available">&#x274C;</span>
              </td>
            </tr>
          </tbody>
        </table>
        <div v-else class="empty-state">
          {{ jobDir ? 'No samples found. Run tracking to populate.' : 'Open an EHT-analytics folder to view job status.' }}
        </div>
      </div>
    </div>

    <!-- Force Visualization -->
    <div class="chart-section" v-if="chartDatasets.length">
      <div class="chart-header">
        <label>Force Visualization</label>
        <button @click="savePlot">Save Plot</button>
      </div>
      <div class="chart-canvas-wrap">
        <Line v-if="chartReady" :data="chartData" :options="chartOptions" ref="chartRef" />
      </div>
    </div>

    <AnalyzeProgressDialog
      :visible="progressVisible"
      :state="progressState"
      @cancel="cancelAnalysis"
      @close="progressVisible = false"
    />
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS,
  LineElement, PointElement, LinearScale, TimeScale, CategoryScale,
  Title, Tooltip, Legend, Filler,
} from 'chart.js'

ChartJS.register(
  LineElement, PointElement, LinearScale, TimeScale, CategoryScale,
  Title, Tooltip, Legend, Filler,
)

import FilePicker from '../shared/FilePicker.vue'
import AnalyzeProgressDialog from './AnalyzeProgressDialog.vue'
import { apiPost } from '../../composables/useApi.js'
import { useTaskProgress } from '../../composables/useTaskProgress.js'

const props = defineProps({
  activeJobDir: { type: String, default: '' },
  visible: { type: Boolean, default: false },
})

const jobDir = ref('')
const samples = ref([])
const selectedIds = ref(new Set())
const errorMsg = ref('')

const { state: progressState, start: startProgress, cancel: cancelProgress } = useTaskProgress()
const progressVisible = ref(false)

// Metadata state
const metadataKeys = ref([])
const addingMetadata = ref(false)
const editingMetaKey = ref(null)     // key currently being edited
const newMetaKey = ref('')
const newMetaValues = ref({})        // { sample_id: value } for adding
const editMetaKey = ref('')          // new key name during edit
const editMetaValues = ref({})       // { sample_id: value } for editing

const canConfirmMeta = computed(() => newMetaKey.value.trim() !== '')

function startAddMetadata() { addingMetadata.value = true; newMetaKey.value = ''; newMetaValues.value = {} }

async function confirmMetadata() {
  const key = newMetaKey.value.trim()
  if (!key) return
  const updates = samples.value.map(a => ({
    id: a.id,
    metadata: { ...(a.metadata || {}), [key]: (newMetaValues.value[a.id] || '').trim() },
  }))
  const keys = [...new Set([...metadataKeys.value, key])]
  try {
    await apiPost('/api/system/save-metadata', { job_dir: jobDir.value, metadata_keys: keys, samples: updates })
    metadataKeys.value = keys
    samples.value = samples.value.map(a => ({
      ...a,
      metadata: { ...(a.metadata || {}), [key]: (newMetaValues.value[a.id] || '').trim() },
    }))
  } catch (e) { errorMsg.value = 'Failed to save metadata: ' + (e.message || '') }
  addingMetadata.value = false; newMetaKey.value = ''; newMetaValues.value = {}
}

function cancelMetadata() { addingMetadata.value = false; newMetaKey.value = ''; newMetaValues.value = {} }

function isMetaKeyBuiltIn(key) { return key === 'Recording' || key === 'ROI Name' }

function startEditMeta(key) {
  editingMetaKey.value = key; editMetaKey.value = key
  editMetaValues.value = {}
  for (const a of samples.value) { editMetaValues.value[a.id] = a.metadata?.[key] || '' }
}

async function confirmEditMeta() {
  const oldKey = editingMetaKey.value
  const newKey = editMetaKey.value.trim()
  if (!newKey || !oldKey) return
  const updates = samples.value.map(a => {
    const meta = { ...(a.metadata || {}) }
    delete meta[oldKey]
    meta[newKey] = (editMetaValues.value[a.id] || '').trim()
    return { id: a.id, metadata: meta }
  })
  const keys = metadataKeys.value.map(k => k === oldKey ? newKey : k)
  try {
    await apiPost('/api/system/save-metadata', { job_dir: jobDir.value, metadata_keys: keys, samples: updates })
    metadataKeys.value = keys
    samples.value = samples.value.map(a => {
      const meta = { ...(a.metadata || {}) }; delete meta[oldKey]; meta[newKey] = (editMetaValues.value[a.id] || '').trim()
      return { ...a, metadata: meta }
    })
  } catch (e) { errorMsg.value = 'Failed to save metadata: ' + (e.message || '') }
  editingMetaKey.value = null; editMetaKey.value = ''; editMetaValues.value = {}
}

function cancelEditMeta() { editingMetaKey.value = null; editMetaKey.value = ''; editMetaValues.value = {} }

async function deleteMeta(key) {
  if (!confirm(`Delete metadata column "${key}" and all its values?`)) return
  const updates = samples.value.map(a => {
    const meta = { ...(a.metadata || {}) }; delete meta[key]; return { id: a.id, metadata: meta }
  })
  const keys = metadataKeys.value.filter(k => k !== key)
  try {
    await apiPost('/api/system/save-metadata', { job_dir: jobDir.value, metadata_keys: keys, samples: updates })
    metadataKeys.value = keys
    samples.value = samples.value.map(a => {
      const meta = { ...(a.metadata || {}) }; delete meta[key]; return { ...a, metadata: meta }
    })
  } catch (e) { errorMsg.value = 'Failed to delete metadata: ' + (e.message || '') }
}

function focusNextMetaInput(event, currentAccId, metaKey) {
  const idx = samples.value.findIndex(a => a.id === currentAccId)
  if (idx < 0 || idx >= samples.value.length - 1) return
  const nextId = samples.value[idx + 1].id
  // Find the input for the next row's same metadata column
  // Use nextTick to let DOM update, then focus
  const selector = `input[data-acc="${nextId}"][data-meta="${metaKey}"]`
  setTimeout(() => {
    const el = document.querySelector(selector)
    if (el) el.focus()
  }, 50)
}
const chartRef = ref(null)
const chartReady = ref(false)
const chartDatasets = ref([])

const COLORS = [
  '#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#9333ea',
  '#0891b2', '#e11d48', '#65a30d', '#d97706', '#4f46e5',
]

const canAnalyze = computed(() =>
  selectedIds.value.size > 0 &&
  samples.value.some(a => selectedIds.value.has(a.id) && a.length_csv)
)

const allSelected = computed(() =>
  samples.value.length > 0 && selectedIds.value.size === samples.value.length
)

const chartData = computed(() => ({
  datasets: chartDatasets.value,
}))

const chartOptions = computed(() => {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { position: 'top', labels: { boxWidth: 16, font: { size: 12 } } },
      tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${parseFloat(ctx.raw.y).toExponential(3)} N` } },
    },
    scales: {
      x: {
        type: 'linear',
        title: { display: true, text: 'Time (s)' },
        min: 0,
      },
      y: {
        title: { display: true, text: 'Force (N)' },
        ticks: { callback: (v) => v.toExponential(2) },
      },
    },
  }
})

watch(() => props.activeJobDir, (dir) => {
  if (dir && dir !== jobDir.value) { jobDir.value = dir; loadConfig() }
})
watch(() => props.visible, (v) => { if (v && jobDir.value) loadConfig() })

// Watch selectedIds — load chart data when checked samples change (only when visible)
watch(selectedIds, () => { if (props.visible) loadChartData() }, { deep: true })

function toggleAll() {
  if (allSelected.value) {
    selectedIds.value = new Set()
  } else {
    selectedIds.value = new Set(samples.value.filter(a => a.length_csv).map(a => a.id))
  }
}

function toggleOne(id) {
  const next = new Set(selectedIds.value)
  if (next.has(id)) { next.delete(id) } else { next.add(id) }
  selectedIds.value = next
}

async function onOpenJobDir(paths) {
  if (!paths[0]) return
  jobDir.value = paths[0]
  await loadConfig()
}

async function loadConfig() {
  errorMsg.value = ''
  try {
    const resp = await apiPost('/api/system/job-config', { job_dir: jobDir.value })
    samples.value = resp.samples || []
    metadataKeys.value = resp.metadata_keys || []
    selectedIds.value = new Set()
    chartDatasets.value = []
  } catch (e) {
    samples.value = []
    errorMsg.value = 'Failed to load config.json: ' + (e.message || 'unknown error')
  }
}

async function loadChartData() {
  const forcePaths = samples.value
    .filter(a => selectedIds.value.has(a.id) && a.force_status_csv)
    .map(a => a.force_status_csv)

  if (!forcePaths.length) {
    chartDatasets.value = []
    chartReady.value = false
    return
  }

  try {
    const resp = await apiPost('/api/analyze/read-csv', {
      job_dir: jobDir.value,
      paths: forcePaths,
    })
    const datasets = []
    let globalMaxT = 0
    const parsed = []

    for (const f of resp.files) {
      const acc = samples.value.find(a => a.force_status_csv && f.filename.endsWith(a.force_status_csv.replace(/\\/g, '/').split('/').pop()))
      const label = acc ? `${acc.id} ${acc.roi_name}` : f.filename
      const cols = f.columns.map(c => c.trim().toLowerCase())
      const tIdx = cols.indexOf('time')
      const fIdx = cols.indexOf('force')
      if (tIdx < 0 || fIdx < 0) continue

      const points = []
      for (const row of f.rows) {
        const t = parseFloat(row[tIdx])
        const force = parseFloat(row[fIdx])
        if (isNaN(t) || isNaN(force)) continue
        points.push({ x: t, y: force })
        if (t > globalMaxT) globalMaxT = t
      }
      if (points.length) {
        parsed.push({ label, points, maxT: points[points.length - 1].x })
      }
    }

    // Use the max time across ALL selected as x-axis range
    for (let i = 0; i < parsed.length; i++) {
      datasets.push({
        label: parsed[i].label,
        data: parsed[i].points,
        borderColor: COLORS[i % COLORS.length],
        backgroundColor: COLORS[i % COLORS.length] + '20',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
        tension: 0,
      })
    }

    chartReady.value = false
    chartDatasets.value = datasets
    await nextTick()
    chartReady.value = true
  } catch (e) {
    errorMsg.value = 'Failed to load force data: ' + (e.message || 'unknown error')
  }
}

function selectedLengthPaths() {
  return samples.value
    .filter(a => selectedIds.value.has(a.id) && a.length_csv)
    .map(a => jobDir.value.replace(/\\/g, '/').replace(/\/$/, '') + '/' + a.length_csv)
}

async function onBatchAnalyze() {
  errorMsg.value = ''
  const paths = selectedLengthPaths()
  if (!paths.length) return

  const existing = []
  try {
    const resp = await apiPost('/api/analyze/validate', { paths })
    existing.push(...resp.valid)
  } catch {}
  if (!existing.length) {
    errorMsg.value = 'None of the selected length CSV files exist. Re-run tracking.'
    return
  }
  if (existing.length < paths.length) {
    errorMsg.value = `${paths.length - existing.length} file(s) not found; analyzing ${existing.length}.`
  }

  progressVisible.value = true
  try {
    const resp = await apiPost('/api/analyze/start', { csv_paths: existing, job_dir: jobDir.value })
    startProgress(resp.task_id, 'analyze', async () => {
      await loadConfig()
    })
  } catch (e) {
    progressState.status = 'failed'
    progressState.errorMessage = e.message
  }
}

function cancelAnalysis() { cancelProgress(); progressVisible.value = false }

function openFile(relPath) {
  apiPost('/api/system/open-folder', { path: jobDir.value, select_file: relPath }).catch(() => {})
}

function savePlot() {
  const canvas = chartRef.value?.chart?.canvas
  if (!canvas) return
  const link = document.createElement('a')
  link.download = 'force_plot.png'
  link.href = canvas.toDataURL('image/png')
  link.click()
}
</script>

<style scoped>
.analyze-root {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 12px;
  gap: 8px;
}

.toolbar {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-shrink: 0;
}

.toolbar :deep(.file-picker-btn) {
  width: 60px;
  flex-shrink: 0;
}

.job-dir-path {
  flex: 1;
  min-width: 0;
  font-size: 12px;
  color: var(--text-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  background: var(--bg);
  padding: 6px 10px;
  border-radius: var(--radius);
  border: 1px solid transparent;
}

.btn-refresh {
  flex-shrink: 0;
}

.btn-refresh:disabled { opacity: 0.4; cursor: not-allowed; }

.selected-count { font-size: 12px; color: var(--primary); font-weight: 500; }

.error-banner {
  padding: 8px 12px;
  background: #fef2f2;
  color: var(--danger);
  font-size: 13px;
  border-radius: var(--radius);
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}
.error-banner button {
  margin-left: auto;
  border: none;
  background: transparent;
  color: var(--danger);
  font-size: 18px;
  padding: 0 4px;
  cursor: pointer;
}

/* ---- Table section ---- */
.table-section {
  flex: 0 0 auto;
  max-height: 40%;
  min-height: 120px;
}
.sample-table-wrapper {
  height: 100%;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: #fff;
}
.sample-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.sample-table th, .sample-table td {
  padding: 6px 10px; text-align: left; border-bottom: 1px solid #f0f0f0; white-space: nowrap;
}
.sample-table th {
  background: #fafafa; font-weight: 600; font-size: 12px; color: var(--text-muted);
  position: sticky; top: 0; z-index: 1;
}
.sample-table tbody tr:hover { background: #f9fafb; }
.sample-table tbody tr.row-selected { background: #eff6ff; }
.col-check { width: 36px; text-align: center; }
.col-check input[type="checkbox"] { cursor: pointer; width: 14px; height: 14px; }
.col-id { font-family: monospace; font-size: 12px; }
.col-meta { font-size: 12px; max-width: 120px; overflow: hidden; text-overflow: ellipsis; }
.col-meta-add { min-width: 80px; text-align: center; }
.col-meta-add button { font-size: 11px; padding: 2px 8px; }
.meta-action { font-size: 12px !important; padding: 0 2px !important; border: none !important; background: transparent !important; cursor: pointer; opacity: 0.5; }
.meta-action:hover { opacity: 1; }
.col-meta-editing { padding: 2px 4px; }
.meta-header-edit { display: flex; gap: 2px; align-items: center; }
.meta-key-input { width: 80px; font-size: 11px; padding: 2px 4px; border: 1px solid var(--border); border-radius: 3px; }
.meta-val-input { width: 80px; font-size: 11px; padding: 2px 4px; border: 1px solid var(--border); border-radius: 3px; }
.btn-confirm, .btn-cancel { border: none; background: transparent; cursor: pointer; font-size: 14px; padding: 0 2px; }
.btn-confirm:disabled { opacity: 0.3; cursor: not-allowed; }
.col-file { text-align: center; width: 80px; }
.check-mark { cursor: pointer; font-size: 16px; user-select: none; }
.check-mark:hover { opacity: 0.7; }
.cross-mark { cursor: default; font-size: 16px; opacity: 0.3; user-select: none; }
.empty-state { padding: 40px; text-align: center; color: var(--text-muted); font-size: 13px; }

/* ---- Chart section ---- */
.chart-section {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: #fff;
  min-height: 200px;
}
.chart-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.chart-header label {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.chart-header button {
  font-size: 12px;
  padding: 4px 12px;
}
.chart-canvas-wrap {
  flex: 1;
  padding: 8px;
  min-height: 0;
}
</style>
