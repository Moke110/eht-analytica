<template>
  <div class="canvas-area" ref="containerRef">
    <canvas
      ref="canvasRef"
      @mousedown="onMouseDown"
      @mousemove="onMouseMove"
      @mouseup="onMouseUp"
      @mouseleave="onMouseUp"
    ></canvas>
    <div
      v-for="(roi, idx) in rois"
      :key="idx"
      class="roi-label"
      :style="roiLabelStyle(roi, idx)"
    >
      <input
        class="roi-name-input"
        :value="roi.name"
        @change="(e) => renameRoi(idx, e.target.value)"
        :style="{ color: rgbStr(roi.color) }"
      />
      <button class="roi-del-btn" @click="deleteRoi(idx)">&times;</button>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, watch, onMounted, onUnmounted, nextTick } from 'vue'

const props = defineProps({
  frameSrc: { type: String, default: '' },
  frameSize: { type: Object, default: () => ({ width: 0, height: 0 }) },
  savedNames: { type: Array, default: () => [] },
})

const emit = defineEmits(['rois-changed', 'names-changed'])

const COLORS = [
  [255, 0, 0], [0, 255, 0], [0, 128, 255],
  [255, 255, 0], [255, 0, 255], [0, 255, 255],
  [255, 128, 0], [255, 128, 255], [128, 255, 128], [255, 192, 203],
]

const canvasRef = ref(null)
const containerRef = ref(null)
const rois = reactive([])

const drawing = ref(false)
const startPos = ref({ x: 0, y: 0 })
const currentRect = ref(null)

let frameImage = null
let canvasW = 0, canvasH = 0
let resizeObserver = null

function rgbStr(color) {
  if (!color) return '#000'
  return `rgb(${color[0]},${color[1]},${color[2]})`
}

function getScaleAndOffset() {
  if (!props.frameSize.width || !props.frameSize.height) return { scale: 1, ox: 0, oy: 0, sw: 0, sh: 0 }
  const scale = Math.min(canvasW / props.frameSize.width, canvasH / props.frameSize.height)
  const sw = props.frameSize.width * scale
  const sh = props.frameSize.height * scale
  return { scale, ox: (canvasW - sw) / 2, oy: (canvasH - sh) / 2, sw, sh }
}

function canvasToPixel(cx, cy, cw, ch) {
  const { scale, ox, oy } = getScaleAndOffset()
  const fw = props.frameSize.width
  const fh = props.frameSize.height
  return {
    x: Math.max(0, Math.min(Math.round((cx - ox) / scale), fw)),
    y: Math.max(0, Math.min(Math.round((cy - oy) / scale), fh)),
    width: Math.max(0, Math.min(Math.round(cw / scale), fw)),
    height: Math.max(0, Math.min(Math.round(ch / scale), fh)),
  }
}

function drawAll() {
  const canvas = canvasRef.value
  if (!canvas) return
  const container = containerRef.value
  const cw = container.clientWidth
  const ch = container.clientHeight
  if (cw <= 0 || ch <= 0) return

  const dpr = window.devicePixelRatio || 1
  canvasW = cw
  canvasH = ch
  canvas.width = cw * dpr
  canvas.height = ch * dpr
  canvas.style.width = cw + 'px'
  canvas.style.height = ch + 'px'

  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  ctx.fillStyle = '#e5e5e5'
  ctx.fillRect(0, 0, cw, ch)

  if (frameImage) {
    const { scale, ox, oy, sw, sh } = getScaleAndOffset()
    ctx.drawImage(frameImage, ox, oy, sw, sh)
  }

  for (const roi of rois) {
    drawRoiRect(ctx, roi)
  }

  if (drawing.value && currentRect.value) {
    const r = currentRect.value
    const color = COLORS[rois.length % COLORS.length]
    ctx.strokeStyle = rgbStr(color)
    ctx.lineWidth = 2
    ctx.setLineDash([6, 3])
    ctx.strokeRect(r.x, r.y, r.width, r.height)
    ctx.setLineDash([])
  }
}

function drawRoiRect(ctx, roi) {
  const { scale, ox, oy } = getScaleAndOffset()
  const cx = roi.x * scale + ox
  const cy = roi.y * scale + oy
  const cw = roi.width * scale
  const ch = roi.height * scale
  ctx.strokeStyle = rgbStr(roi.color)
  ctx.lineWidth = 2
  ctx.setLineDash([])
  ctx.strokeRect(cx, cy, cw, ch)
}

// Mouse handlers
function onMouseDown(e) {
  const rect = canvasRef.value.getBoundingClientRect()
  const x = e.clientX - rect.left
  const y = e.clientY - rect.top
  drawing.value = true
  startPos.value = { x, y }
  currentRect.value = null
}

function onMouseMove(e) {
  if (!drawing.value) return
  const rect = canvasRef.value.getBoundingClientRect()
  const cx = e.clientX - rect.left
  const cy = e.clientY - rect.top
  const x = Math.min(startPos.value.x, cx)
  const y = Math.min(startPos.value.y, cy)
  const w = Math.abs(cx - startPos.value.x)
  const h = Math.abs(cy - startPos.value.y)
  currentRect.value = { x, y, width: w, height: h }
  drawAll()
}

function onMouseUp() {
  if (!drawing.value) return
  drawing.value = false
  const r = currentRect.value
  currentRect.value = null

  if (r && r.width > 10 && r.height > 10) {
    const pixel = canvasToPixel(r.x, r.y, r.width, r.height)
    const color = COLORS[rois.length % COLORS.length]
    const idx = rois.length
    const name = props.savedNames[idx] || defaultRoiName(idx + 1)
    rois.push({
      name,
      x: pixel.x,
      y: pixel.y,
      width: pixel.width,
      height: pixel.height,
      color: [...color],
    })
    emit('rois-changed', rois.length)
    emitNames()
  }
  drawAll()
}

function renameRoi(idx, newName) {
  if (!newName.trim()) return
  if (rois.some((r, i) => i !== idx && r.name === newName.trim())) return
  rois[idx].name = newName.trim()
  emitNames()
}

function deleteRoi(idx) {
  rois.splice(idx, 1)
  emit('rois-changed', rois.length)
  emitNames()
  drawAll()
}

function getRois() {
  return rois.map(r => ({
    name: r.name,
    x: r.x,
    y: r.y,
    width: r.width,
    height: r.height,
    color: r.color,
  }))
}

function getNames() {
  return rois.map(r => r.name)
}

function defaultRoiName(n) {
  return `EHT-${n}`
}

function emitNames() {
  emit('names-changed', getNames())
}

function clearRois() {
  rois.splice(0, rois.length)
  emit('rois-changed', 0)
  emitNames()
  drawAll()
}

function roiLabelStyle(roi) {
  const { scale, ox, oy } = getScaleAndOffset()
  const cx = roi.x * scale + ox
  const cy = roi.y * scale + oy
  const ch = roi.height * scale
  const labelH = 28
  const hasSpaceAbove = cy >= labelH + 4
  return {
    left: `${cx}px`,
    top: hasSpaceAbove ? `${cy - labelH - 2}px` : `${cy + ch + 2}px`,
  }
}

watch(() => props.frameSrc, async (src) => {
  if (!src) return
  frameImage = new Image()
  frameImage.onload = () => nextTick(drawAll)
  frameImage.src = `data:image/png;base64,${src}`
})

onMounted(() => {
  if (containerRef.value) {
    resizeObserver = new ResizeObserver(() => drawAll())
    resizeObserver.observe(containerRef.value)
  }
  nextTick(drawAll)
})

onUnmounted(() => {
  if (resizeObserver) {
    resizeObserver.disconnect()
    resizeObserver = null
  }
})

defineExpose({ getRois, clearRois })
</script>

<style scoped>
.canvas-area {
  width: 100%;
  height: 100%;
  position: relative;
  background: #e5e5e5;
  overflow: hidden;
}

canvas {
  display: block;
}

.roi-label {
  position: absolute;
  display: flex;
  gap: 4px;
  align-items: center;
  background: rgba(255,255,255,0.92);
  border: 1px solid rgba(0,0,0,0.12);
  border-radius: 4px;
  padding: 2px 4px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  z-index: 10;
  white-space: nowrap;
}

.roi-name-input {
  border: none;
  background: transparent;
  font-weight: 600;
  font-size: 12px;
  width: 90px;
  outline: none;
  padding: 2px 4px;
  border-radius: 2px;
}

.roi-name-input:focus {
  background: rgba(0,0,0,0.04);
}

.roi-del-btn {
  border: none;
  background: transparent;
  color: #dc2626;
  font-size: 16px;
  font-weight: bold;
  cursor: pointer;
  padding: 0 2px;
  line-height: 1;
  border-radius: 50%;
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.roi-del-btn:hover {
  background: rgba(220,38,38,0.1);
}
</style>
