<template>
  <div class="reports-root">
    <!-- Toolbar -->
    <div class="reports-toolbar">
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
        @click="loadData"
        title="Refresh table from config.json"
      >Refresh</button>
      <span class="selected-count" v-if="selectedIds.size">{{ selectedIds.size }} selected</span>
    </div>

    <!-- Sample table - max 10 visible rows -->
    <div class="reports-table-wrap" v-if="filteredSamples.length">
      <table class="reports-table">
        <thead>
          <tr>
            <th class="col-check"><input type="checkbox" :checked="allSelected" @change="toggleAll" /></th>
            <th>Sample ID</th>
            <th>Recording</th>
            <th>ROI Name</th>
            <th v-for="mk in metadataKeys" :key="mk">{{ mk }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="acc in filteredSamples" :key="acc.id" :class="{ selected: selectedIds.has(acc.id) }">
            <td class="col-check">
              <input type="checkbox" :checked="selectedIds.has(acc.id)" @change="toggleOne(acc.id)" />
            </td>
            <td class="col-id">{{ acc.id }}</td>
            <td>{{ acc.recording_name }}</td>
            <td>{{ acc.roi_name }}</td>
            <td v-for="mk in metadataKeys" :key="mk">{{ resolveMeta(acc, mk) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <div v-else class="empty-state">No samples with metrics available.</div>

    <!-- Grouping selectors -->
    <div class="grouping-bar">
      <span class="group-label">X group by:</span>
      <button
        v-for="mk in allGroupKeys"
        :key="'x-'+mk"
        class="group-tag"
        :class="{ active: groupKeys.has(mk) }"
        @click="toggleGroupKey(mk)"
      >{{ mk }}</button>
      <span v-if="!allGroupKeys.length" class="group-hint">No grouping dimensions available</span>
    </div>
    <div class="grouping-bar" v-if="groupKeys.size > 0">
      <span class="group-label">Dot group by:</span>
      <button
        v-for="mk in allGroupKeys"
        :key="'dot-'+mk"
        class="group-tag"
        :class="{ active: dotGroupKeys.has(mk), disabled: groupKeys.has(mk) }"
        @click="toggleDotGroupKey(mk)"
        :disabled="groupKeys.has(mk)"
      >{{ mk }}</button>
    </div>
    <div class="grouping-bar">
      <span class="spacer"></span>
      <button class="btn-save" :disabled="!canSave" @click="onSavePlots">Save plots</button>
      <button class="btn-save" :disabled="!canSave" @click="onSaveData">Save plot CSVs</button>
      <button class="btn-save" :disabled="!hasSelection" @click="onSaveFullData">Save full data</button>
    </div>

    <!-- Plots - always visible -->
    <ReportsPlot
      ref="reportsPlotRef"
      :job-dir="jobDir"
      :sample-ids="selectedIds"
      :group-keys="groupKeys"
      :dot-group-keys="dotGroupKeys"
    />
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import { apiPost } from '../../composables/useApi.js'
import FilePicker from '../shared/FilePicker.vue'
import ReportsPlot from './ReportsPlot.vue'

const props = defineProps({
  activeJobDir: { type: String, default: '' },
})

const jobDir = ref('')
const samples = ref([])
const metadataKeys = ref([])
const selectedIds = ref(new Set())
const groupKeys = ref(new Set())
const dotGroupKeys = ref(new Set())
const reportsPlotRef = ref(null)

const canSave = computed(() =>
  reportsPlotRef.value && selectedIds.value.size > 0 &&
  (groupKeys.value.size > 0 || dotGroupKeys.value.size > 0)
)

const hasSelection = computed(() => selectedIds.value.size > 0)

const filteredSamples = computed(() =>
  samples.value.filter(a => a.metrics_csv)
)

const allSelected = computed(() =>
  filteredSamples.value.length > 0 && selectedIds.value.size === filteredSamples.value.length
)

watch(() => props.activeJobDir, (dir) => { if (dir) { jobDir.value = dir; loadData() } })
onMounted(() => { if (props.activeJobDir) { jobDir.value = props.activeJobDir; loadData() } })

function toggleAll() {
  if (allSelected.value) selectedIds.value = new Set()
  else selectedIds.value = new Set(filteredSamples.value.map(a => a.id))
}
function toggleOne(id) {
  const n = new Set(selectedIds.value); n.has(id) ? n.delete(id) : n.add(id); selectedIds.value = n
}
function toggleGroupKey(k) {
  const next = new Set(groupKeys.value)
  if (next.has(k)) { next.delete(k) }
  else {
    next.add(k)
    // Deselect from dot keys if present
    if (dotGroupKeys.value.has(k)) {
      const dotNext = new Set(dotGroupKeys.value); dotNext.delete(k); dotGroupKeys.value = dotNext
    }
  }
  groupKeys.value = next
}

function toggleDotGroupKey(k) {
  if (groupKeys.value.has(k)) return  // disabled
  const next = new Set(dotGroupKeys.value)
  if (next.has(k)) { next.delete(k) } else { next.add(k) }
  dotGroupKeys.value = next
}

function onOpenJobDir(paths) {
  if (!paths[0]) return
  jobDir.value = paths[0]
  loadData()
}

async function loadData() {
  try {
    const resp = await apiPost('/api/system/job-config', { job_dir: jobDir.value })
    samples.value = resp.samples || []
    const md = resp.metadata_keys || []
    metadataKeys.value = md
    groupKeys.value = new Set()
    dotGroupKeys.value = new Set()
    selectedIds.value = new Set()
  } catch { samples.value = [] }
}

const allGroupKeys = computed(() => {
  const mk = metadataKeys.value
  const extra = []
  if (!mk.includes('Recording')) extra.push('Recording')
  if (!mk.includes('ROI Name')) extra.push('ROI Name')
  return [...extra, ...mk]
})

function resolveMeta(acc, key) {
  if (key === 'Recording') return acc.recording_name || ''
  if (key === 'ROI Name') return acc.roi_name || ''
  return (acc.metadata && acc.metadata[key]) || ''
}

async function onSavePlots() {
  const dir = await pickSaveDir('reports_plots_dir')
  if (!dir) return
  const charts = reportsPlotRef.value?.chartRefs
  if (!charts) return
  const baseName = jobDir.value.split(/[\\/]/).filter(Boolean).pop() || 'reports'
  for (const [metric, ref] of Object.entries(charts)) {
    const canvas = ref?.chart?.canvas
    if (!canvas) continue
    const safeName = metric.replace(/[/\\?%*:|"<>]/g, '_')
    const dataUrl = canvas.toDataURL('image/png')
    await apiPost('/api/system/save-file', {
      dir, filename: `${baseName}_${safeName}.png`, content: dataUrl,
    }).catch(() => {})
  }
}

async function onSaveData() {
  const dir = await pickSaveDir('reports_data_dir')
  if (!dir) return
  const exportData = reportsPlotRef.value?.getPlotDataForExport()
  if (!exportData) return
  const csv = reportsPlotRef.value.generateCsvContent(exportData)
  const baseName = jobDir.value.split(/[\\/]/).filter(Boolean).pop() || 'reports'
  await apiPost('/api/system/save-file', {
    dir, filename: `${baseName}_report.csv`, content: csv,
  }).catch(() => {})
}

async function onSaveFullData() {
  const dir = await pickSaveDir('reports_data_dir')
  if (!dir) return
  const ids = [...selectedIds.value]
  if (!ids.length) return

  // Fetch per-sample metrics from backend
  let fullData
  try {
    const resp = await apiPost('/api/system/reports/full-data', {
      job_dir: jobDir.value,
      sample_ids: ids,
    })
    fullData = resp
  } catch (e) {
    console.error('Failed to fetch full data:', e)
    return
  }

  const baseName = jobDir.value.split(/[\\/]/).filter(Boolean).pop() || 'reports'
  const samples = fullData.samples || []
  const allMetrics = fullData.metrics || []
  const allMetaKeys = fullData.metadata_keys || []

  // Columns: Sample ID, Recording, ROI Name, metadata keys..., metric names...
  const cols = ['Sample ID', 'Recording', 'ROI Name', ...allMetaKeys, ...allMetrics]
  const lines = [cols.join(',')]

  for (const s of samples) {
    const row = [
      s.id,
      csvCell(s.recording_name),
      csvCell(s.roi_name),
      ...allMetaKeys.map(k => csvCell((s.metadata || {})[k] || '')),
      ...allMetrics.map(m => {
        const v = (s.metrics || {})[m]
        return v != null ? v : ''
      }),
    ]
    lines.push(row.join(','))
  }

  const csv = lines.join('\n')
  await apiPost('/api/system/save-file', {
    dir, filename: `${baseName}_full_data.csv`, content: csv,
  }).catch(() => {})
}

function csvCell(val) {
  const s = String(val ?? '')
  if (s.includes(',') || s.includes('"') || s.includes('\n')) {
    return `"${s.replace(/"/g, '""')}"`
  }
  return s
}

async function pickSaveDir(key) {
  return apiPost('/api/system/config/load-path', { key })
    .then(r => r.path || '')
    .catch(() => '')
    .then(savedPath => apiPost('/api/system/native-file-dialog', {
      type: 'directory',
      title: 'Select directory to save',
      initial_path: savedPath,
    }))
    .then(r => {
      const dir = r.paths?.[0]
      if (dir) apiPost('/api/system/config/save-path', { key, path: dir }).catch(() => {})
      return dir || ''
    })
    .catch(() => '')
}
</script>

<style scoped>
.reports-root {
  display: flex;
  flex-direction: column;
  padding: 12px;
  gap: 10px;
}
.reports-toolbar {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-shrink: 0;
}
.reports-toolbar :deep(.file-picker-btn) { width: 60px; flex-shrink: 0; }
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
}
.selected-count { font-size: 12px; color: var(--primary); font-weight: 500; flex-shrink: 0; }

.reports-table-wrap {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: #fff;
  max-height: 340px;
  overflow-y: auto;
}
.reports-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.reports-table th, .reports-table td {
  padding: 5px 8px;
  text-align: left;
  border-bottom: 1px solid #f0f0f0;
  white-space: nowrap;
}
.reports-table th {
  background: #fafafa;
  font-weight: 600;
  font-size: 12px;
  color: var(--text-muted);
  position: sticky;
  top: 0;
  z-index: 1;
}
.reports-table tr.selected { background: #eff6ff; }
.col-check { width: 30px; text-align: center; }
.col-id { font-variant-numeric: tabular-nums; }
.col-check input { cursor: pointer; width: 14px; height: 14px; }
.empty-state { padding: 16px; text-align: center; color: var(--text-muted); font-size: 13px; }

.grouping-bar {
  display: flex;
  gap: 6px;
  align-items: center;
  flex-shrink: 0;
  flex-wrap: wrap;
  padding: 4px 0;
}
.group-label { font-size: 12px; font-weight: 600; color: var(--text-muted); }
.group-tag {
  font-size: 11px;
  padding: 3px 10px;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: #fff;
  cursor: pointer;
}
.group-tag.active { background: var(--primary); color: #fff; border-color: var(--primary); }
.group-tag.disabled { background: #f5f5f5; color: #bbb; cursor: not-allowed; border-color: #e0e0e0; }
.group-hint { font-size: 11px; color: var(--text-muted); }
.spacer { flex: 1; }
.btn-save { font-size: 12px; padding: 4px 12px; }
.btn-save:disabled { opacity: 0.4; cursor: not-allowed; }
</style>
