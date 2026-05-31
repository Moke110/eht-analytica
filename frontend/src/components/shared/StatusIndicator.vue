<template>
  <span class="status-indicator" :class="status">
    <span class="status-text">{{ displayText }}</span>
    <span v-if="status === 'running'" class="dots">{{ dots }}</span>
  </span>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  statusText: { type: String, default: '' },
  status: { type: String, default: 'idle' }, // idle | running | ready | error
})

const dots = ref('')
let timer = null

onMounted(() => {
  timer = setInterval(() => {
    dots.value = dots.value.length >= 3 ? '' : dots.value + '.'
  }, 500)
})

onUnmounted(() => {
  clearInterval(timer)
})

const displayText = computed(() => {
  if (props.status === 'ready') return props.statusText || 'Ready'
  if (props.status === 'error') return props.statusText || 'Error'
  return props.statusText
})
</script>
