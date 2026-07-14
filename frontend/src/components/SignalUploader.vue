<!-- SignalUploader.vue
     Lets the user choose a built-in demo sample OR upload a CSV file.
-->
<template>
  <div class="bg-white border border-gray-200 rounded-xl p-5 space-y-4">
    <h2 class="text-sm font-semibold text-gray-700 uppercase tracking-wide">Input Signal</h2>

    <!-- Source toggle -->
    <div class="flex rounded-lg overflow-hidden border border-gray-200 text-sm font-medium">
      <button
        class="flex-1 py-2 transition-colors"
        :class="mode === 'sample' ? 'bg-blue-600 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'"
        @click="mode = 'sample'"
      >Demo Sample</button>
      <button
        class="flex-1 py-2 transition-colors border-l border-gray-200"
        :class="mode === 'upload' ? 'bg-blue-600 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'"
        @click="mode = 'upload'"
      >Upload CSV</button>
    </div>

    <!-- Demo sample picker -->
    <div v-if="mode === 'sample'" class="space-y-2">
      <label class="block text-xs text-gray-500">Select a PhysioNet MIT-BIH record</label>
      <select
        v-model="selectedSampleId"
        class="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-400"
      >
        <option disabled value="">— choose a record —</option>
        <option v-for="s in samples" :key="s.id" :value="s.id">
          {{ s.id }} — {{ s.label }}
        </option>
      </select>
    </div>

    <!-- CSV upload -->
    <div v-else class="space-y-2">
      <label class="block text-xs text-gray-500">One column of mV values (no header required)</label>
      <div
        class="border-2 border-dashed border-gray-200 rounded-lg p-6 text-center cursor-pointer hover:border-blue-400 transition-colors"
        @click="fileInput?.click()"
        @dragover.prevent
        @drop.prevent="onDrop"
      >
        <p v-if="!uploadedFile" class="text-sm text-gray-400">
          Click or drag &amp; drop a .csv file
        </p>
        <p v-else class="text-sm text-blue-600 font-medium">{{ uploadedFile.name }}</p>
      </div>
      <input ref="fileInput" type="file" accept=".csv" class="hidden" @change="onFileChange" />
    </div>

    <!-- Options row -->
    <div class="grid grid-cols-2 gap-3">
      <div>
        <label class="block text-xs text-gray-500 mb-1">Model</label>
        <select v-model="modelType" class="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-400">
          <option value="rf">Random Forest</option>
          <option value="xgb">XGBoost</option>
        </select>
      </div>
      <div>
        <label class="block text-xs text-gray-500 mb-1">Sampling rate (Hz)</label>
        <input
          v-model.number="samplingRate"
          type="number"
          min="100" max="1000"
          class="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-400"
        />
      </div>
    </div>

    <div class="flex items-center gap-3">
      <label class="flex items-center gap-2 cursor-pointer text-sm text-gray-600">
        <input type="checkbox" v-model="includeLlm" class="rounded" />
        AI explanation (requires Ollama)
      </label>
    </div>

    <!-- Classify button -->
    <button
      class="w-full py-2.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      :disabled="loading || !canClassify"
      @click="classify"
    >
      <span v-if="loading">Classifying…</span>
      <span v-else>Classify ECG</span>
    </button>

    <!-- Error -->
    <p v-if="error" class="text-sm text-red-500">{{ error }}</p>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { fetchSamples, fetchSample, classifySignal, classifyUpload } from '../api/client.js'

const emit = defineEmits(['result'])

const mode = ref('sample')
const samples = ref([])
const selectedSampleId = ref('')
const uploadedFile = ref(null)
const modelType = ref('rf')
const samplingRate = ref(360)
const includeLlm = ref(false)
const loading = ref(false)
const error = ref('')
const fileInput = ref(null)

const canClassify = computed(() =>
  mode.value === 'upload' ? !!uploadedFile.value : !!selectedSampleId.value
)

onMounted(async () => {
  try {
    samples.value = await fetchSamples()
    if (samples.value.length) selectedSampleId.value = samples.value[0].id
  } catch {
    error.value = 'Could not connect to backend. Make sure the API is running.'
  }
})

function onFileChange(e) {
  uploadedFile.value = e.target.files[0] ?? null
}
function onDrop(e) {
  uploadedFile.value = e.dataTransfer.files[0] ?? null
}

async function classify() {
  error.value = ''
  loading.value = true
  try {
    let result
    if (mode.value === 'sample') {
      const record = await fetchSample(selectedSampleId.value)
      result = await classifySignal(record.signal, {
        samplingRate: record.sampling_rate,
        modelType: modelType.value,
        includeLlm: includeLlm.value,
      })
    } else {
      result = await classifyUpload(uploadedFile.value, {
        samplingRate: samplingRate.value,
        modelType: modelType.value,
        includeLlm: includeLlm.value,
      })
    }
    emit('result', result)
  } catch (err) {
    error.value = err.response?.data?.detail ?? err.message ?? 'Classification failed.'
  } finally {
    loading.value = false
  }
}
</script>
