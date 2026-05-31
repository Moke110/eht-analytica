<template>
  <div class="csv-list">
    <div v-if="paths.length === 0" class="csv-empty">No CSV files added</div>
    <div
      v-for="(path, idx) in paths"
      :key="idx"
      class="csv-item"
      :class="{ selected: selected.has(idx) }"
    >
      <input
        type="checkbox"
        :checked="selected.has(idx)"
        @change="toggle(idx)"
      />
      <span>{{ path }}</span>
    </div>
  </div>
  <div class="csv-count">{{ paths.length }} file(s) added</div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  paths: { type: Array, default: () => [] },
})

const emit = defineEmits(['update:paths'])

const selected = ref(new Set())

function toggle(idx) {
  if (selected.value.has(idx)) {
    selected.value.delete(idx)
  } else {
    selected.value.add(idx)
  }
  selected.value = new Set(selected.value)
}

function removeSelected() {
  const keep = props.paths.filter((_, i) => !selected.value.has(i))
  selected.value = new Set()
  emit('update:paths', keep)
}

function clearAll() {
  selected.value = new Set()
  emit('update:paths', [])
}

defineExpose({ removeSelected, clearAll })
</script>

<style scoped>
.csv-empty {
  padding: 20px;
  text-align: center;
  color: var(--text-muted);
  font-size: 13px;
}
</style>
