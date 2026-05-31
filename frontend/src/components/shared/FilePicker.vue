<template>
  <button class="file-picker-btn" @click="pickFiles" :disabled="disabled">
    {{ buttonLabel }}
  </button>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { apiGet, apiPost } from '../../composables/useApi.js'

const props = defineProps({
  buttonLabel: { type: String, default: 'Select...' },
  dialogType: { type: String, default: 'file' },
  title: { type: String, default: 'Select' },
  filters: { type: Array, default: () => [] },
  multi: { type: Boolean, default: false },
  disabled: { type: Boolean, default: false },
  pickerKey: { type: String, default: '' },
})

const emit = defineEmits(['paths-selected'])

const initialPath = ref('')

onMounted(async () => {
  if (!props.pickerKey) return
  try {
    const config = await apiGet('/api/system/config')
    initialPath.value = config[props.pickerKey] || ''
  } catch {}
})

async function pickFiles() {
  try {
    const resp = await apiPost('/api/system/native-file-dialog', {
      type: props.dialogType,
      title: props.title,
      filters: props.filters,
      multi: props.multi,
      initial_path: initialPath.value || undefined,
    })
    if (resp.paths && resp.paths.length > 0) {
      // Save the directory for next time
      if (props.pickerKey) {
        apiPost('/api/system/config/save-path', {
          key: props.pickerKey,
          path: resp.paths[0],
        }).catch(() => {})
      }
      emit('paths-selected', resp.paths)
    }
  } catch (e) {
    console.error('File dialog error:', e)
  }
}
</script>
