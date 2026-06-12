<template>
  <Teleport to="body">
    <div class="modal-backdrop" v-if="visible">
      <div class="modal-box">
        <h3>{{ title }}</h3>
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
          <button
            v-if="state.status !== 'running'"
            @click="$emit('close')"
          >Close</button>
        </div>

        <div v-if="state.status === 'failed'" class="error-box">
          <div class="error-box-title">Tracking Failed</div>
          <div class="error-box-body">{{ state.errorMessage }}</div>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup>
defineProps({
  visible: { type: Boolean, default: false },
  title: { type: String, default: 'Processing...' },
  state: { type: Object, default: () => ({ percent: 0, message: '', status: 'running' }) },
})

defineEmits(['cancel', 'close'])
</script>

<style scoped>
.progress-msg { font-size: 13px; color: var(--text-muted); margin-bottom: 4px; }
.progress-pct { font-size: 12px; color: var(--text-muted); text-align: right; }
.error-box {
  margin-top: 16px;
  padding: 12px 14px;
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 6px;
}
.error-box-title {
  font-size: 14px;
  font-weight: 600;
  color: #b91c1c;
  margin-bottom: 6px;
}
.error-box-body {
  font-size: 13px;
  color: var(--danger);
  word-break: break-word;
  white-space: pre-wrap;
  max-height: 120px;
  overflow-y: auto;
}
</style>
