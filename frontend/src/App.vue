<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Header -->
    <header class="bg-white border-b border-gray-200 px-6 py-4 flex items-center gap-3">
      <svg class="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" d="M3 12h2l2-7 4 14 3-10 2 5h5" />
      </svg>
      <h1 class="text-lg font-bold text-gray-800">ECG Arrhythmia Classifier</h1>
      <span class="ml-auto text-xs text-gray-400">MIT-BIH · Random Forest · SHAP</span>
    </header>

    <!-- Tab bar -->
    <div class="bg-white border-b border-gray-200 px-6 flex gap-0">
      <button
        v-for="tab in TABS" :key="tab.id"
        class="px-5 py-3 text-sm font-medium border-b-2 transition-colors"
        :class="activeTab === tab.id
          ? 'border-blue-600 text-blue-600'
          : 'border-transparent text-gray-500 hover:text-gray-700'"
        @click="activeTab = tab.id"
      >{{ tab.label }}</button>
    </div>

    <main class="max-w-7xl mx-auto px-4 py-6">

      <!-- ── Classify tab ── -->
      <div v-show="activeTab === 'classify'" class="grid grid-cols-1 lg:grid-cols-[340px_1fr] gap-6">
        <!-- Left: input controls -->
        <aside>
          <SignalUploader @result="onResult" />
        </aside>

        <!-- Right: results -->
        <section class="space-y-5">
          <!-- Empty state -->
          <div
            v-if="!result"
            class="bg-white border border-dashed border-gray-200 rounded-xl flex items-center justify-center"
            style="min-height: 420px;"
          >
            <div class="text-center text-gray-400 space-y-2">
              <svg class="w-12 h-12 mx-auto opacity-30" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M3 12h2l2-7 4 14 3-10 2 5h5" />
              </svg>
              <p class="text-sm">Select a sample or upload a CSV and click <strong>Classify ECG</strong></p>
            </div>
          </div>

          <template v-else>
            <!-- Waveform with beat window overlay -->
            <ECGWaveform
              :signal="result.filtered_signal"
              :rpeaks="result.rpeaks"
              :beats="result.beats"
              :sampling-rate="result.sampling_rate"
              :highlight-beat="selectedBeat"
              @select-beat="selectedBeat = $event"
            />

            <!-- Bottom row: classification + SHAP -->
            <div class="grid grid-cols-1 xl:grid-cols-2 gap-5">
              <ClassificationResult
                :result="result"
                :selected-beat="selectedBeat"
                @select-beat="selectedBeat = $event"
              />
              <SHAPChart
                :features="selectedBeatFeatures"
                :beat-index="selectedBeat"
              />
            </div>

            <!-- Top features text breakdown -->
            <div v-if="selectedBeatObj" class="bg-white border border-gray-200 rounded-xl p-4">
              <p class="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
                Beat {{ selectedBeat }}
                <span class="ml-1 px-2 py-0.5 rounded-full text-white text-xs"
                      :style="{ background: labelColor(selectedBeatObj.label) }">
                  {{ selectedBeatObj.label }}
                </span>
                — Top Contributing Features
              </p>
              <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <div
                  v-for="f in selectedBeatFeatures"
                  :key="f.name"
                  class="flex items-center justify-between bg-gray-50 rounded-lg px-3 py-2 text-sm"
                >
                  <span class="text-gray-700 font-medium">{{ f.name }}</span>
                  <span
                    class="font-mono text-xs font-semibold"
                    :class="f.direction === 'positive' ? 'text-red-500' : 'text-blue-500'"
                  >{{ f.shap >= 0 ? '+' : '' }}{{ f.shap.toFixed(4) }}</span>
                </div>
              </div>
              <!-- LLM explanation -->
              <div v-if="result.llm_explanation" class="mt-4 bg-amber-50 border border-amber-200 rounded-lg p-3">
                <p class="text-xs font-semibold text-amber-600 mb-1">AI Explanation</p>
                <p class="text-sm text-gray-700 leading-relaxed">{{ result.llm_explanation }}</p>
              </div>
            </div>
          </template>
        </section>
      </div>

      <!-- ── Validation tab ── -->
      <div v-show="activeTab === 'validation'">
        <ValidationPanel />
      </div>

    </main>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import SignalUploader from './components/SignalUploader.vue'
import ECGWaveform from './components/ECGWaveform.vue'
import ClassificationResult from './components/ClassificationResult.vue'
import SHAPChart from './components/SHAPChart.vue'
import ValidationPanel from './components/ValidationPanel.vue'

const TABS = [
  { id: 'classify', label: 'Classify' },
  { id: 'validation', label: 'Validation Report' },
]
const activeTab = ref('classify')

const result = ref(null)
const selectedBeat = ref(null)

const LABEL_COLORS = { N:'#22c55e', AF:'#f97316', PVC:'#ef4444', OTHER:'#64748b' }
const labelColor = (l) => LABEL_COLORS[l] ?? '#64748b'

function onResult(data) {
  result.value = data
  selectedBeat.value = data.beats.length > 0 ? 0 : null
}

const selectedBeatObj = computed(() =>
  result.value?.beats?.find(b => b.beat_index === selectedBeat.value) ?? null
)
const selectedBeatFeatures = computed(() => selectedBeatObj.value?.top_features ?? [])
</script>
