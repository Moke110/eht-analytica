<template>
  <div class="app">
    <header class="tab-bar">
      <button
        class="tab-btn"
        :class="{ active: activeTab === 'track' }"
        @click="activeTab = 'track'"
      >Track</button>
      <button
        class="tab-btn"
        :class="{ active: activeTab === 'analyze' }"
        @click="activeTab = 'analyze'"
      >Analyze</button>
      <span class="app-title">EHT Analytica</span>
      <span class="model-badge" v-if="modelLoaded">{{ modelName }}</span>
    </header>
    <main class="tab-content">
      <div v-if="modelError" class="model-error-banner">
        Model load failed: {{ modelError }}
        <button @click="modelError = ''">&times;</button>
      </div>
      <TrackPanel v-if="activeTab === 'track'" />
      <AnalyzePanel v-else-if="activeTab === 'analyze'" />
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import TrackPanel from './components/track/TrackPanel.vue'
import AnalyzePanel from './components/analyze/AnalyzePanel.vue'
import { apiGet, apiPost } from './composables/useApi.js'

const activeTab = ref('track')
const modelLoaded = ref(false)
const modelName = ref('')
const modelError = ref('')

onMounted(async () => {
  try {
    const config = await apiGet('/api/system/config')
    if (config.track_model_path) {
      const resp = await apiPost('/api/track/model/load', { path: config.track_model_path })
      const es = new EventSource(`/api/track/task/${resp.task_id}/stream`)
      es.onmessage = (e) => {
        if (!e.data) return
        try {
          const d = JSON.parse(e.data)
          if (d.event === 'complete') {
            modelLoaded.value = true
            modelName.value = d.result?.model_name || ''
            es.close()
          } else if (d.event === 'error') {
            modelError.value = d.message
            es.close()
          }
        } catch {}
      }
      es.onerror = () => es.close()
    }
  } catch {
    // unable to reach backend — retry will happen on user action
  }
})
</script>
