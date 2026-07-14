<!-- ValidationPanel.vue
     Shows per-class F1, confusion matrix and patient-level results from training.
     Fetched from GET /api/validation on mount.
-->
<template>
  <div class="bg-white border border-gray-200 rounded-xl p-5 space-y-5">
    <h2 class="text-sm font-semibold text-gray-700 uppercase tracking-wide">
      Model Validation Report
      <span class="ml-2 text-xs font-normal text-gray-400 normal-case">Patient-held-out test set</span>
    </h2>

    <div v-if="loading" class="text-sm text-gray-400 text-center py-6">Loading validation results…</div>
    <div v-else-if="error" class="text-sm text-red-500">{{ error }}</div>

    <template v-else-if="data">
      <!-- Patient split summary -->
      <div class="grid grid-cols-3 gap-3">
        <div class="bg-gray-50 border border-gray-200 rounded-lg p-3 text-center">
          <p class="text-xs text-gray-400">Train patients</p>
          <p class="text-xl font-bold text-gray-800">{{ data.train_patients?.length ?? '—' }}</p>
        </div>
        <div class="bg-gray-50 border border-gray-200 rounded-lg p-3 text-center">
          <p class="text-xs text-gray-400">Test patients</p>
          <p class="text-xl font-bold text-gray-800">{{ data.test_patients?.length ?? '—' }}</p>
        </div>
        <div class="bg-gray-50 border border-gray-200 rounded-lg p-3 text-center">
          <p class="text-xs text-gray-400">Patient accuracy</p>
          <p class="text-xl font-bold" :class="patAccColor">
            {{ patientAccuracy !== null ? (patientAccuracy * 100).toFixed(1) + '%' : '—' }}
          </p>
        </div>
      </div>

      <!-- Per-class metrics table -->
      <div>
        <p class="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Per-class metrics</p>
        <table class="w-full text-sm">
          <thead class="bg-gray-50">
            <tr>
              <th class="px-3 py-2 text-left text-xs text-gray-500 font-medium rounded-tl-lg">Class</th>
              <th class="px-3 py-2 text-right text-xs text-gray-500 font-medium">Precision</th>
              <th class="px-3 py-2 text-right text-xs text-gray-500 font-medium">Recall</th>
              <th class="px-3 py-2 text-right text-xs text-gray-500 font-medium">F1</th>
              <th class="px-3 py-2 text-right text-xs text-gray-500 font-medium rounded-tr-lg">Support</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="row in perClass"
              :key="row.label"
              class="border-t border-gray-100"
            >
              <td class="px-3 py-2">
                <span class="px-2 py-0.5 rounded-full text-xs font-semibold text-white"
                      :style="{ background: labelColor(row.label) }">{{ row.label }}</span>
              </td>
              <td class="px-3 py-2 text-right text-gray-600">{{ pct(row.precision) }}</td>
              <td class="px-3 py-2 text-right text-gray-600">{{ pct(row.recall) }}</td>
              <td class="px-3 py-2 text-right font-semibold" :class="f1Color(row.f1)">{{ pct(row.f1) }}</td>
              <td class="px-3 py-2 text-right text-gray-400">{{ row.support }}</td>
            </tr>
          </tbody>
          <tfoot class="border-t-2 border-gray-200 bg-gray-50">
            <tr>
              <td class="px-3 py-2 text-xs text-gray-500 font-semibold">Macro F1</td>
              <td colspan="3" class="px-3 py-2 text-right font-bold text-gray-700">{{ pct(macroF1) }}</td>
              <td class="px-3 py-2 text-right text-xs text-gray-400">{{ totalBeats }}</td>
            </tr>
          </tfoot>
        </table>
      </div>

      <!-- Confusion matrix -->
      <div v-if="confMatrix">
        <p class="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
          Confusion Matrix <span class="font-normal normal-case text-gray-400">(rows=true, cols=predicted)</span>
        </p>
        <div class="overflow-x-auto">
          <table class="text-xs border-collapse">
            <thead>
              <tr>
                <th class="w-10"></th>
                <th
                  v-for="lbl in confMatrix.labels"
                  :key="lbl"
                  class="px-2 py-1 text-center font-semibold"
                  :style="{ color: labelColor(lbl) }"
                >{{ lbl }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, ri) in confMatrix.matrix" :key="ri">
                <td class="pr-2 py-1 text-right font-semibold"
                    :style="{ color: labelColor(confMatrix.labels[ri]) }">
                  {{ confMatrix.labels[ri] }}
                </td>
                <td
                  v-for="(val, ci) in row"
                  :key="ci"
                  class="px-2 py-1 text-center rounded font-mono"
                  :class="ri === ci ? 'font-bold' : ''"
                  :style="cellStyle(val, ri, ci)"
                >{{ val }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p class="text-xs text-gray-400 mt-1">Green diagonal = correct. Off-diagonal = errors.</p>
      </div>

      <!-- R-peak validation -->
      <div v-if="rpValidation">
        <p class="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
          R-peak Detection Validation (Pan-Tompkins vs cardiologist annotations)
        </p>
        <div class="grid grid-cols-3 gap-2">
          <div class="bg-gray-50 border border-gray-200 rounded-lg p-2 text-center">
            <p class="text-xs text-gray-400">Mean sensitivity</p>
            <p class="text-lg font-bold text-green-600">{{ pct(rpValidation.mean_sensitivity) }}</p>
          </div>
          <div class="bg-gray-50 border border-gray-200 rounded-lg p-2 text-center">
            <p class="text-xs text-gray-400">Mean PPV</p>
            <p class="text-lg font-bold text-blue-600">{{ pct(rpValidation.mean_ppv) }}</p>
          </div>
          <div class="bg-gray-50 border border-gray-200 rounded-lg p-2 text-center">
            <p class="text-xs text-gray-400">Records</p>
            <p class="text-lg font-bold text-gray-700">{{ rpValidation.n_records }}</p>
          </div>
        </div>
      </div>

    </template>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { fetchValidation } from '../api/client.js'

const loading = ref(true)
const error = ref('')
const data = ref(null)

const LABEL_COLORS = { N:'#22c55e', AF:'#f97316', PVC:'#ef4444', OTHER:'#64748b' }
const labelColor = (l) => LABEL_COLORS[l] ?? '#64748b'
const pct = (v) => v != null ? (v * 100).toFixed(1) + '%' : '—'
const f1Color = (f1) => f1 >= 0.8 ? 'text-green-600' : f1 >= 0.6 ? 'text-amber-500' : 'text-red-500'

const perClass = computed(() => data.value?.test_metrics?.per_class ?? [])
const macroF1 = computed(() => data.value?.test_metrics?.macro_f1 ?? null)
const totalBeats = computed(() => data.value?.n_test_beats ?? '—')
const confMatrix = computed(() => data.value?.confusion_matrix ?? null)
const patientAccuracy = computed(() => data.value?.patient_report?.patient_accuracy ?? null)
const patAccColor = computed(() =>
  patientAccuracy.value >= 0.8 ? 'text-green-600' :
  patientAccuracy.value >= 0.6 ? 'text-amber-500' : 'text-red-500'
)

const rpValidation = computed(() => {
  const rv = data.value?.rpeak_validation
  if (!rv || !Object.keys(rv).length) return null
  const vals = Object.values(rv)
  return {
    n_records: vals.length,
    mean_sensitivity: vals.reduce((s, v) => s + v.sensitivity, 0) / vals.length,
    mean_ppv: vals.reduce((s, v) => s + v.ppv, 0) / vals.length,
  }
})

function cellStyle(val, ri, ci) {
  if (!confMatrix.value) return {}
  const row = confMatrix.value.matrix[ri]
  const rowMax = Math.max(...row, 1)
  const frac = val / rowMax
  if (ri === ci) return { background: `rgba(34,197,94,${Math.max(0.08, frac * 0.5)})` }
  if (val > 0) return { background: `rgba(239,68,68,${Math.min(frac * 0.6, 0.4)})` }
  return {}
}

onMounted(async () => {
  try {
    data.value = await fetchValidation()
  } catch (e) {
    error.value = e.response?.status === 404
      ? 'No validation data found. Run python train.py first.'
      : (e.message ?? 'Failed to load validation results.')
  } finally {
    loading.value = false
  }
})
</script>
