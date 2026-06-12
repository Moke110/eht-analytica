<template>
  <div class="reports-plot-root">
    <div v-if="!ready && !hasPrevData" class="plot-placeholder">
      Select samples and group keys to generate plots
    </div>

    <div v-show="ready || hasPrevData" class="charts-area">
      <div v-if="loading" class="plot-overlay">Loading&hellip;</div>
      <div v-if="error && !loading" class="plot-overlay plot-overlay-error">{{ error }}</div>
      <div v-if="!loading && !error && ready && !plotData.metrics.length" class="plot-overlay">
        No data for the current selection
      </div>

      <div class="charts-grid" v-show="plotData.metrics.length > 0">
        <div v-for="metric in plotData.metrics" :key="metric" class="chart-card">
          <div class="chart-title">{{ metric }}</div>
          <div class="chart-canvas-wrap">
            <Bar v-if="!isLineMode" :ref="el => setChartRef(metric, el)" :data="buildChartData(metric)" :options="buildChartOptions(metric)" :plugins="[dotsErrorbarPlugin]" />
            <Line v-else :ref="el => setChartRef(metric, el)" :data="buildChartData(metric)" :options="buildChartOptions(metric)" :plugins="[lineErrorbarPlugin]" />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { Bar, Line } from 'vue-chartjs'
import {
  Chart,
  BarElement, BarController,
  LineElement, LineController, PointElement,
  CategoryScale, LinearScale,
  Tooltip, Filler,
} from 'chart.js'
import { apiPost } from '../../composables/useApi.js'

Chart.register(BarElement, BarController, LineElement, LineController, PointElement, CategoryScale, LinearScale, Tooltip, Filler)

const EDGE_COLORS = [
  '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
  '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD',
]

/* ---- HSV color helpers ---- */
function hexToRgb(hx) {
  return {
    r: parseInt(hx.slice(1, 3), 16) / 255,
    g: parseInt(hx.slice(3, 5), 16) / 255,
    b: parseInt(hx.slice(5, 7), 16) / 255,
  }
}
function rgbToHsv(r, g, b) {
  const max = Math.max(r, g, b), min = Math.min(r, g, b)
  const d = max - min
  let h = 0
  if (d !== 0) {
    if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6
    else if (max === g) h = ((b - r) / d + 2) / 6
    else h = ((r - g) / d + 4) / 6
  }
  const s = max === 0 ? 0 : d / max
  return { h: h * 360, s, v: max }
}
function hsvToRgb(h, s, v) {
  const c = v * s
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1))
  const m = v - c
  let r, g, b
  if (h < 60) { r = c; g = x; b = 0 }
  else if (h < 120) { r = x; g = c; b = 0 }
  else if (h < 180) { r = 0; g = c; b = x }
  else if (h < 240) { r = 0; g = x; b = c }
  else if (h < 300) { r = x; g = 0; b = c }
  else { r = c; g = 0; b = x }
  return { r: r + m, g: g + m, b: b + m }
}
function rgbToHex(r, g, b) {
  const toHex = (v) => Math.round(Math.max(0, Math.min(1, v)) * 255).toString(16).padStart(2, '0').toUpperCase()
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`
}

function fillColor(edgeHex) {
  const { r, g, b } = hexToRgb(edgeHex)
  const { h, s, v } = rgbToHsv(r, g, b)
  const ns = s * 0.40
  const nv = Math.min(v * 1.30, 1.0)
  const rgb = hsvToRgb(h, ns, nv)
  return rgbToHex(rgb.r, rgb.g, rgb.b)
}

function dotColor(edgeHex) {
  const { r, g, b } = hexToRgb(edgeHex)
  const { h, s, v } = rgbToHsv(r, g, b)
  const ns = s * 0.65
  const nv = Math.min(v * 1.15, 1.0)
  const rgb = hsvToRgb(h, ns, nv)
  return rgbToHex(rgb.r, rgb.g, rgb.b)
}

const COLOR_TRIPLES = EDGE_COLORS.map((edge) => ({
  edge,
  fill: fillColor(edge),
  dot: dotColor(edge),
}))

const props = defineProps({
  jobDir: { type: String, default: '' },
  sampleIds: { type: Set, default: () => new Set() },
  groupKeys: { type: Set, default: () => new Set() },
  dotGroupKeys: { type: Set, default: () => new Set() },
})

const plotData = ref({ metrics: [], groups: [], data: {} })
const loading = ref(false)
const error = ref(null)
const hasPrevData = ref(false)
const lastFetchGood = ref(false)  // true when the last doFetch returned real data

const isLineMode = computed(() => {
  return props.dotGroupKeys && props.dotGroupKeys.size > 0 && props.groupKeys && props.groupKeys.size > 0
})

const ready = computed(() => {
  return props.jobDir && props.sampleIds.size > 0 && props.groupKeys && props.groupKeys.size > 0
})

const chartRefs = {}

function setChartRef(metric, el) {
  if (el) chartRefs[metric] = el
  else delete chartRefs[metric]
}

function getPlotDataForExport() {
  const pd = plotData.value
  if (!pd || !pd.metrics.length) return null

  if (isLineMode.value) {
    // In line mode: rows = metrics, columns = X_dot_label pairs
    const xKeys = [...props.groupKeys]
    const dotKeys = [...props.dotGroupKeys]
    const groups = []
    for (const x of pd.xLabels) {
      for (const d of pd.dotLabels) {
        groups.push(`${x}_${d}`)
      }
    }
    const rows = []
    for (const metric of pd.metrics) {
      const row = { Metric: metric }
      for (const g of groups) {
        const gm = pd.lineData[metric]?.[g]
        row[g] = gm && gm.n > 0 ? gm.mean.toExponential(6) : ''
      }
      rows.push(row)
    }
    return { metrics: pd.metrics, groups, rows }
  }

  // Bar mode
  const rows = []
  for (const metric of pd.metrics) {
    const row = { Metric: metric }
    const metricData = pd.data[metric] || {}
    for (const group of pd.groups) {
      const gm = metricData[group]
      row[group] = gm && gm.sample_values && gm.sample_values.length > 0
        ? gm.mean.toExponential(6) : ''
    }
    rows.push(row)
  }
  return { metrics: pd.metrics, groups: pd.groups, rows }
}

function generateCsvContent(exportData) {
  const cols = ['Metric', ...exportData.groups]
  const lines = [cols.join(',')]
  for (const row of exportData.rows) {
    lines.push(cols.map(c => {
      const v = row[c]
      if (v === undefined || v === null) return ''
      if (typeof v === 'string' && (v.includes(',') || v.includes('"') || v.includes('\n'))) {
        return `"${v.replace(/"/g, '""')}"`
      }
      return String(v)
    }).join(','))
  }
  return lines.join('\n')
}

defineExpose({ chartRefs, getPlotDataForExport, generateCsvContent, plotData })

// ── Resolve a metadata label for a sample ────────────────────────
function resolveLabel(sample, key) {
  if (key === 'Recording') return sample.recording_name || ''
  if (key === 'ROI Name') return sample.roi_name || ''
  return (sample.metadata && sample.metadata[key]) || ''
}

let debounceTimer = null
let abortController = null

function doFetch() {
  abortController?.abort()
  abortController = new AbortController()

  const ids = [...props.sampleIds]

  if (!ready.value) {
    error.value = null
    loading.value = false
    return
  }

  loading.value = true
  error.value = null

  if (isLineMode.value) {
    fetchLineData(ids)
  } else {
    fetchBarData(ids)
  }
}

function fetchBarData(ids) {
  const keys = [...props.groupKeys]

  apiPost('/api/system/reports/plot-data', { job_dir: props.jobDir, sample_ids: ids, group_keys: keys }, abortController.signal)
    .then(data => {
      plotData.value = data
      lastFetchGood.value = data.metrics.length > 0
      if (data.metrics.length > 0) hasPrevData.value = true
    })
    .catch(err => {
      if (err?.name === 'AbortError') return
      error.value = err.message || 'Failed to load plot data'
      if (!lastFetchGood.value) plotData.value = { metrics: [], groups: [], data: {} }
    })
    .finally(() => { loading.value = false })
}

function fetchLineData(ids) {
  const xKeys = [...props.groupKeys]
  const dotKeys = [...props.dotGroupKeys]

  apiPost('/api/system/reports/full-data', { job_dir: props.jobDir, sample_ids: ids }, abortController.signal)
    .then(data => {
      const result = computeLinePlotData(data, xKeys, dotKeys)
      plotData.value = result
      lastFetchGood.value = result.metrics.length > 0
      if (result.metrics.length > 0) hasPrevData.value = true
    })
    .catch(err => {
      if (err?.name === 'AbortError') return
      error.value = err.message || 'Failed to load data'
      if (!lastFetchGood.value) plotData.value = { metrics: [], groups: [], data: {} }
    })
    .finally(() => { loading.value = false })
}

function computeLinePlotData(fullData, xKeys, dotKeys) {
  const samples = fullData.samples || []
  const metricNames = fullData.metrics || []

  // Group samples by (xLabel, dotLabel)
  // For each group, collect metric values per metric
  const setMap = {}  // key: "xLabel___dotLabel" -> { xLabel, dotLabel, samples: [], metrics: {metric: [values]} }
  const xLabelSet = new Set()
  const dotLabelSet = new Set()

  for (const s of samples) {
    const xLabel = xKeys.map(k => resolveLabel(s, k)).join('_')
    const dotLabel = dotKeys.map(k => resolveLabel(s, k)).join('_')
    const key = `${xLabel}___${dotLabel}`
    xLabelSet.add(xLabel)
    dotLabelSet.add(dotLabel)

    if (!setMap[key]) {
      setMap[key] = { xLabel, dotLabel, metrics: {} }
    }
    const cell = setMap[key]
    for (const m of metricNames) {
      if (s.metrics && s.metrics[m] != null) {
        if (!cell.metrics[m]) cell.metrics[m] = []
        cell.metrics[m].push(s.metrics[m])
      }
    }
  }

  const xLabels = [...xLabelSet].sort()
  const dotLabels = [...dotLabelSet].sort()

  // Build per-metric line data: { "xLabel___dotLabel": { mean, ste, n, sample_values } }
  const lineData = {}
  for (const m of metricNames) {
    lineData[m] = {}
    for (const [key, set] of Object.entries(setMap)) {
      const vals = set.metrics[m]
      if (!vals || vals.length === 0) {
        lineData[m][key] = { mean: 0, ste: 0, n: 0, sample_values: [] }
        continue
      }
      const n = vals.length
      const mean = vals.reduce((a, b) => a + b, 0) / n
      let ste = 0
      if (n > 1) {
        const variance = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1)
        ste = Math.sqrt(variance) / Math.sqrt(n)
      }
      lineData[m][key] = { mean, ste, n, sample_values: vals }
    }
  }

  return {
    metrics: metricNames,
    xLabels,
    dotLabels,
    lineData,
    groups: xLabels,      // for save compatibility
    data: lineData,        // for save compatibility
  }
}

// ── Watcher ─────────────────────────────────────────────────────
watch(
  () => [props.jobDir, [...props.sampleIds], [...props.groupKeys], [...(props.dotGroupKeys || [])]],
  (newVals, oldVals) => {
    if (oldVals && newVals[0] !== oldVals[0]) {
      hasPrevData.value = false
      lastFetchGood.value = false
      plotData.value = { metrics: [], groups: [], data: {} }
    }
    clearTimeout(debounceTimer)
    debounceTimer = setTimeout(doFetch, 300)
  },
  { deep: true, immediate: true }
)

// ══════════════════════════════════════════════════════════════════
//  BAR MODE — dots + error bars plugin
// ══════════════════════════════════════════════════════════════════
const dotsErrorbarPlugin = {
  id: 'dotsErrorbar',
  afterDraw(chart) {
    const meta = chart.getDatasetMeta(0)
    if (!meta || !meta.data) return
    const groups = chart.options.plugins?.dotsErrorbar?.groups
    if (!groups || !groups.length) return

    const ctx = chart.ctx
    const yScale = chart.scales.y

    groups.forEach((cfg, i) => {
      const barEl = meta.data[i]
      if (!barEl) return

      const cx = barEl.x
      const bw = barEl.width || 30

      const dColor = COLOR_TRIPLES[i % COLOR_TRIPLES.length].dot
      ;(cfg.sample_values || []).forEach((val, vi) => {
        const y = yScale.getPixelForValue(val)
        const jitter = (((Math.abs(val) * 10000 + vi * 7) % 100) / 100 - 0.5) * bw * 0.6
        ctx.beginPath()
        ctx.arc(cx + jitter, y, 3.5, 0, Math.PI * 2)
        ctx.fillStyle = dColor
        ctx.fill()
      })

      if (!cfg.ste || cfg.ste <= 0) return
      const topY = yScale.getPixelForValue(cfg.mean - cfg.ste)
      const botY = yScale.getPixelForValue(cfg.mean + cfg.ste)
      ctx.save()
      ctx.strokeStyle = '#000'
      ctx.lineWidth = 1.5
      ctx.beginPath(); ctx.moveTo(cx, topY); ctx.lineTo(cx, botY); ctx.stroke()
      const cap = 6
      ctx.beginPath(); ctx.moveTo(cx - cap, topY); ctx.lineTo(cx + cap, topY); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(cx - cap, botY); ctx.lineTo(cx + cap, botY); ctx.stroke()
      ctx.restore()
    })
  },
}

// ══════════════════════════════════════════════════════════════════
//  LINE MODE — square dots + error bars plugin
// ══════════════════════════════════════════════════════════════════
const SQ_HALF = 5  // half-size of square dot

const lineErrorbarPlugin = {
  id: 'lineErrorbar',
  afterDraw(chart) {
    const ctx = chart.ctx
    const yScale = chart.scales.y
    const xScale = chart.scales.x
    const metasets = chart.getSortedVisibleDatasetMetas()

    metasets.forEach((meta, dsIdx) => {
      const ds = chart.data.datasets[dsIdx]
      if (!ds || !ds._setMeta) return  // _setMeta: array of {ste, n} per data point

      const color = EDGE_COLORS[dsIdx % EDGE_COLORS.length]
      ctx.save()

      meta.data.forEach((el, i) => {
        if (!el) return
        const sm = ds._setMeta[i]
        if (!sm || sm.n < 2 || !sm.ste || sm.ste <= 0) return

        const cx = el.x
        const topY = yScale.getPixelForValue(sm.mean + sm.ste)
        const botY = yScale.getPixelForValue(sm.mean - sm.ste)

        // Error bar
        ctx.strokeStyle = '#000'
        ctx.lineWidth = 1.5
        ctx.beginPath(); ctx.moveTo(cx, topY); ctx.lineTo(cx, botY); ctx.stroke()
        const cap = 6
        ctx.beginPath(); ctx.moveTo(cx - cap, topY); ctx.lineTo(cx + cap, topY); ctx.stroke()
        ctx.beginPath(); ctx.moveTo(cx - cap, botY); ctx.lineTo(cx + cap, botY); ctx.stroke()

        // Square dot (drawn on top to cover line intersection)
        ctx.fillStyle = color
        ctx.strokeStyle = color
        ctx.lineWidth = 1.5
        const dotY = yScale.getPixelForValue(sm.mean)
        ctx.fillRect(cx - SQ_HALF, dotY - SQ_HALF, SQ_HALF * 2, SQ_HALF * 2)
      })

      ctx.restore()
    })
  },
}

// ══════════════════════════════════════════════════════════════════
//  Chart builders
// ══════════════════════════════════════════════════════════════════
function buildChartData(metric) {
  const pd = plotData.value
  if (!pd) return { labels: [], datasets: [] }

  if (isLineMode.value) return buildLineChartData(metric, pd)
  return buildBarChartData(metric, pd)
}

function buildBarChartData(metric, pd) {
  const groups = pd.groups || []
  const metricData = pd.data[metric] || {}

  return {
    labels: groups,
    datasets: [{
      data: groups.map(g => {
        const gm = metricData[g]
        if (!gm || !gm.sample_values || gm.sample_values.length === 0) return null
        return gm.mean ?? 0
      }),
      backgroundColor: groups.map((_, i) => COLOR_TRIPLES[i % COLOR_TRIPLES.length].fill),
      borderColor: groups.map((_, i) => COLOR_TRIPLES[i % COLOR_TRIPLES.length].edge),
      borderWidth: 1.5,
      barPercentage: 0.6,
      categoryPercentage: 0.8,
    }],
  }
}

function buildLineChartData(metric, pd) {
  const xLabels = pd.xLabels || []
  const dotLabels = pd.dotLabels || []
  const metricLineData = pd.lineData[metric] || {}

  const datasets = dotLabels.map((dotLabel, di) => {
    const color = EDGE_COLORS[di % EDGE_COLORS.length]
    const dataPoints = xLabels.map(xLabel => {
      const key = `${xLabel}___${dotLabel}`
      const cell = metricLineData[key]
      if (!cell || cell.n === 0) return null
      return { x: xLabel, y: cell.mean }
    })

    return {
      label: dotLabel,
      data: dataPoints,
      borderColor: color,
      backgroundColor: color,
      borderWidth: 2,
      pointRadius: 0,       // dots drawn by plugin
      pointHitRadius: 8,
      fill: false,
      tension: 0,
      spanGaps: false,
      // Store set metadata for the errorbar plugin
      _setMeta: xLabels.map(xLabel => {
        const key = `${xLabel}___${dotLabel}`
        return metricLineData[key] || { mean: 0, ste: 0, n: 0 }
      }),
    }
  })

  return {
    labels: xLabels,
    datasets,
  }
}

function buildChartOptions(metric) {
  const pd = plotData.value
  if (!pd) return {}

  if (isLineMode.value) return buildLineChartOptions(metric, pd)
  return buildBarChartOptions(metric, pd)
}

function buildBarChartOptions(metric, pd) {
  const groups = pd.groups || []
  const metricData = pd.data[metric] || {}

  let yMax = 0
  groups.forEach(g => {
    const gm = metricData[g]
    if (!gm) return
    ;(gm.sample_values || []).forEach(v => { yMax = Math.max(yMax, v) })
    if (gm.ste > 0) yMax = Math.max(yMax, gm.mean + gm.ste)
  })

  return {
    responsive: true,
    maintainAspectRatio: true,
    aspectRatio: 5 / 4,
    animation: { duration: 600, easing: 'easeOutQuart' },
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label(ctx) {
            const cfg = metricData[ctx.label]
            if (!cfg) return ''
            return [`mean: ${cfg.mean}`, `STE: ${cfg.ste}`, `n: ${cfg.sample_values.length}`]
          },
        },
      },
      dotsErrorbar: {
        groups: groups.map(g => metricData[g] || { sample_values: [], mean: 0, ste: 0 }),
      },
    },
    scales: {
      x: { grid: { display: false } },
      y: {
        beginAtZero: true,
        max: yMax > 0 ? yMax * 1.08 : undefined,
        title: { display: true, text: metric },
      },
    },
  }
}

function buildLineChartOptions(metric, pd) {
  const dotLabels = pd.dotLabels || []
  const metricLineData = pd.lineData[metric] || {}

  let yMax = 0
  for (const key of Object.keys(metricLineData)) {
    const cell = metricLineData[key]
    if (!cell) continue
    ;(cell.sample_values || []).forEach(v => { yMax = Math.max(yMax, v) })
    if (cell.ste > 0) yMax = Math.max(yMax, cell.mean + cell.ste)
  }

  return {
    responsive: true,
    maintainAspectRatio: true,
    aspectRatio: 5 / 4,
    animation: { duration: 400, easing: 'easeOutQuart' },
    interaction: { mode: 'nearest', intersect: true },
    plugins: {
      legend: {
        display: dotLabels.length > 1,
        position: 'top',
        labels: { boxWidth: 14, font: { size: 11 }, usePointStyle: true, pointStyle: 'rect' },
      },
      tooltip: {
        callbacks: {
          label(ctx) {
            const ds = ctx.chart.data.datasets[ctx.datasetIndex]
            const sm = ds._setMeta?.[ctx.dataIndex]
            if (!sm || sm.n === 0) return `${ctx.dataset.label}: N/A`
            return [`${ctx.dataset.label}: ${sm.mean.toExponential(4)}`, `STE: ${sm.ste.toExponential(4)}`, `n: ${sm.n}`]
          },
        },
      },
    },
    scales: {
      x: {
        type: 'category',
        title: { display: true, text: props.groupKeys ? [...props.groupKeys].join(' + ') : '' },
        grid: { display: false },
      },
      y: {
        beginAtZero: true,
        max: yMax > 0 ? yMax * 1.12 : undefined,
        title: { display: true, text: metric },
      },
    },
  }
}
</script>

<style scoped>
.reports-plot-root {
  padding-top: 4px;
}
.plot-placeholder {
  font-size: 13px;
  color: var(--text-muted);
  text-align: center;
  padding: 32px 0;
}
.charts-area {
  position: relative;
}
.plot-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  color: var(--text-muted);
  background: rgba(255,255,255,0.85);
  z-index: 2;
  pointer-events: none;
}
.plot-overlay-error {
  color: #c44e52;
}
.charts-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}
.chart-card {
  background: #fff;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 8px;
  display: flex;
  flex-direction: column;
}
.chart-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-muted);
  text-align: center;
  margin-bottom: 4px;
}
.chart-canvas-wrap {
  width: 100%;
}
</style>
