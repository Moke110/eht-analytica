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

        <div v-if="state.status === 'failed'" class="error-msg">
          {{ state.errorMessage }}
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
.error-msg { font-size: 12px; color: var(--danger); margin-top: 8px; word-break: break-word; }
</style>
