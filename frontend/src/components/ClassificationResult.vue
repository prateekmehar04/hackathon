<!-- ClassificationResult.vue
     Displays the dominant rhythm badge, confidence, beat list, and LLM explanation.
-->
<template>
  <div class="space-y-4">
    <!-- Dominant rhythm card -->
    <div class="bg-white border border-gray-200 rounded-xl p-5 flex items-center gap-5">
      <div
        class="w-16 h-16 rounded-full flex items-center justify-center text-white text-lg font-bold shrink-0"
        :style="{ background: labelColor(result.dominant_label) }"
      >
        {{ result.dominant_label }}
      </div>
      <div>
        <p class="text-xs text-gray-400 uppercase tracking-wide mb-0.5">Dominant Rhythm</p>
        <p class="text-xl font-semibold text-gray-800">{{ labelDescription(result.dominant_label) }}</p>
        <div class="flex items-center gap-2 mt-1">
          <div class="flex-1 bg-gray-100 rounded-full h-2 w-40">
            <div
              class="h-2 rounded-full transition-all duration-500"
              :style="{ width: `${result.dominant_confidence * 100}%`, background: labelColor(result.dominant_label) }"
            ></div>
          </div>
          <span class="text-sm font-medium text-gray-600">
            {{ (result.dominant_confidence * 100).toFixed(1) }}%
          </span>
        </div>
      </div>
    </div>

    <!-- Beat table -->
    <div class="bg-white border border-gray-200 rounded-xl overflow-hidden">
      <div class="px-4 py-3 border-b border-gray-100 flex items-center justify-between">
        <h2 class="text-sm font-semibold text-gray-700 uppercase tracking-wide">Beat-by-Beat Results</h2>
        <span class="text-xs text-gray-400">{{ result.beats.length }} beats</span>
      </div>
      <div class="overflow-y-auto" style="max-height: 240px;">
        <table class="w-full text-sm">
          <thead class="bg-gray-50 sticky top-0">
            <tr>
              <th class="px-4 py-2 text-left text-xs text-gray-500 font-medium">#</th>
              <th class="px-4 py-2 text-left text-xs text-gray-500 font-medium">Sample</th>
              <th class="px-4 py-2 text-left text-xs text-gray-500 font-medium">Class</th>
              <th class="px-4 py-2 text-left text-xs text-gray-500 font-medium">Confidence</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="beat in result.beats"
              :key="beat.beat_index"
              class="border-t border-gray-50 cursor-pointer transition-colors"
              :class="selectedBeat === beat.beat_index ? 'bg-blue-50' : 'hover:bg-gray-50'"
              @click="$emit('select-beat', beat.beat_index)"
            >
              <td class="px-4 py-2 text-gray-500">{{ beat.beat_index }}</td>
              <td class="px-4 py-2 text-gray-500">{{ beat.rpeak_sample }}</td>
              <td class="px-4 py-2">
                <span
                  class="px-2 py-0.5 rounded-full text-xs font-semibold text-white"
                  :style="{ background: labelColor(beat.label) }"
                >{{ beat.label }}</span>
              </td>
              <td class="px-4 py-2 text-gray-600">{{ (beat.confidence * 100).toFixed(1) }}%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- LLM explanation -->
    <div v-if="result.llm_explanation" class="bg-amber-50 border border-amber-200 rounded-xl p-4">
      <p class="text-xs font-semibold text-amber-600 uppercase tracking-wide mb-2">AI Explanation</p>
      <p class="text-sm text-gray-700 leading-relaxed">{{ result.llm_explanation }}</p>
    </div>
  </div>
</template>

<script setup>
defineProps({
  result: { type: Object, required: true },
  selectedBeat: { type: Number, default: null },
})
defineEmits(['select-beat'])

const LABEL_COLORS = {
  N:     '#22c55e',
  AF:    '#f97316',
  PVC:   '#ef4444',
  TACHY: '#a855f7',
  OTHER: '#64748b',
}

const LABEL_DESCRIPTIONS = {
  N:     'Normal Sinus Rhythm',
  AF:    'Atrial Fibrillation',
  PVC:   'Premature Ventricular Contraction',
  TACHY: 'Tachycardia',
  OTHER: 'Other / Unknown',
}

function labelColor(label) {
  return LABEL_COLORS[label] ?? '#64748b'
}

function labelDescription(label) {
  return LABEL_DESCRIPTIONS[label] ?? label
}
</script>
