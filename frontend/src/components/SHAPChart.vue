<!-- SHAPChart.vue
     Horizontal bar chart of SHAP feature contributions for a single beat.
-->
<template>
  <div class="bg-white border border-gray-200 rounded-xl p-4">
    <h2 class="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">
      Feature Importance (SHAP) — Beat {{ beatIndex }}
    </h2>

    <div v-if="features.length === 0" class="text-sm text-gray-400 text-center py-6">
      Select a beat to see its explanation.
    </div>

    <div v-else style="height: 220px;">
      <canvas ref="canvasRef"></canvas>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, onBeforeUnmount } from 'vue'
import {
  Chart, BarController, BarElement, CategoryScale,
  LinearScale, Tooltip,
} from 'chart.js'

Chart.register(BarController, BarElement, CategoryScale, LinearScale, Tooltip)

const props = defineProps({
  features: { type: Array, default: () => [] },   // top_features array from API
  beatIndex: { type: Number, default: null },
})

const canvasRef = ref(null)
let chart = null

function buildChart() {
  if (!canvasRef.value || props.features.length === 0) return
  if (chart) { chart.destroy(); chart = null }

  const sorted = [...props.features].sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
  const labels = sorted.map(f => f.name)
  const values = sorted.map(f => f.shap)
  const colors = values.map(v => v >= 0 ? 'rgba(239,68,68,0.75)' : 'rgba(59,130,246,0.75)')

  chart = new Chart(canvasRef.value, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'SHAP value',
        data: values,
        backgroundColor: colors,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: 'y',
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (item) => {
              const f = sorted[item.dataIndex]
              return ` SHAP=${f.shap.toFixed(4)}  val=${f.value.toFixed(3)}`
            },
          },
        },
      },
      scales: {
        x: {
          grid: { color: '#f1f5f9' },
          ticks: { color: '#94a3b8', font: { size: 10 } },
        },
        y: {
          ticks: { color: '#374151', font: { size: 11 } },
          grid: { display: false },
        },
      },
    },
  })
}

onMounted(buildChart)
watch(() => props.features, buildChart, { deep: true })
onBeforeUnmount(() => chart?.destroy())
</script>
