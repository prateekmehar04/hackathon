<!-- ECGWaveform.vue
     Renders the ECG strip using Chart.js with:
       - R-peak scatter points coloured by classification label
       - Beat window highlight overlay (box annotation via chartjs-plugin-annotation)
       - Click-to-select beat
-->
<template>
  <div class="bg-white border border-gray-200 rounded-xl p-4">
    <div class="flex items-center justify-between mb-3">
      <h2 class="text-sm font-semibold text-gray-700 uppercase tracking-wide">ECG Waveform</h2>
      <div class="flex items-center gap-3">
        <span v-if="samplingRate" class="text-xs text-gray-400">{{ samplingRate }} Hz</span>
        <span v-if="highlightBeat !== null" class="text-xs text-blue-500 font-medium">
          Beat {{ highlightBeat }} selected
        </span>
      </div>
    </div>

    <div class="relative" style="height: 240px;">
      <canvas ref="canvasRef"></canvas>
    </div>

    <!-- Legend -->
    <div class="flex flex-wrap gap-3 mt-3">
      <div v-for="(color, label) in LABEL_COLORS" :key="label" class="flex items-center gap-1">
        <span class="inline-block w-3 h-3 rounded-full" :style="{ background: color }"></span>
        <span class="text-xs text-gray-500">{{ label }}</span>
      </div>
      <div class="flex items-center gap-1 ml-2">
        <span class="inline-block w-5 h-2 rounded" style="background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.4)"></span>
        <span class="text-xs text-gray-500">Beat window</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, onBeforeUnmount } from 'vue'
import {
  Chart, LineController, LineElement, PointElement,
  LinearScale, CategoryScale, ScatterController, Tooltip, Filler,
} from 'chart.js'

Chart.register(
  LineController, LineElement, PointElement,
  LinearScale, CategoryScale, ScatterController, Tooltip, Filler,
)

const props = defineProps({
  signal: { type: Array, default: () => [] },
  rpeaks: { type: Array, default: () => [] },
  beats: { type: Array, default: () => [] },       // full beat objects from API
  samplingRate: { type: Number, default: 360 },
  highlightBeat: { type: Number, default: null },
})

const LABEL_COLORS = {
  N:     '#22c55e',
  AF:    '#f97316',
  PVC:   '#ef4444',
  OTHER: '#64748b',
}

const DOWNSAMPLE = 4
const canvasRef = ref(null)
let chart = null

function buildChart() {
  if (!canvasRef.value || props.signal.length === 0) return
  if (chart) { chart.destroy(); chart = null }

  const sig = props.signal
  const ds = DOWNSAMPLE

  // Downsample display signal
  const sigData = []
  for (let i = 0; i < sig.length; i += ds) sigData.push({ x: i, y: sig[i] })

  // R-peak scatter
  const rpData = props.rpeaks.map((idx, beatIdx) => ({
    x: Math.round(idx / ds),
    y: sig[idx] ?? 0,
    beatIdx,
    label: props.beats[beatIdx]?.label ?? 'N',
  }))

  // Beat window background regions drawn via custom plugin
  const beatWindows = props.beats.map((b, i) => ({
    xMin: Math.round((b.beat_start ?? 0) / ds),
    xMax: Math.round((b.beat_end ?? 0) / ds),
    color: i === props.highlightBeat
      ? 'rgba(59,130,246,0.18)'
      : 'rgba(200,210,220,0.10)',
    border: i === props.highlightBeat
      ? 'rgba(59,130,246,0.5)'
      : 'rgba(200,210,220,0)',
  }))

  // Inline annotation plugin (no external dep required)
  const beatOverlayPlugin = {
    id: 'beatOverlay',
    beforeDraw(chart) {
      const { ctx, chartArea: { left, right, top, bottom }, scales } = chart
      if (!scales.x) return
      ctx.save()
      beatWindows.forEach(w => {
        const x1 = scales.x.getPixelForValue(w.xMin)
        const x2 = scales.x.getPixelForValue(w.xMax)
        if (x2 < left || x1 > right) return
        ctx.fillStyle = w.color
        ctx.fillRect(Math.max(x1, left), top, Math.min(x2, right) - Math.max(x1, left), bottom - top)
        if (w.border !== 'rgba(200,210,220,0)') {
          ctx.strokeStyle = w.border
          ctx.lineWidth = 1.5
          ctx.setLineDash([3, 3])
          ctx.strokeRect(Math.max(x1, left), top, Math.min(x2, right) - Math.max(x1, left), bottom - top)
          ctx.setLineDash([])
        }
      })
      ctx.restore()
    },
  }

  chart = new Chart(canvasRef.value, {
    type: 'scatter',
    plugins: [beatOverlayPlugin],
    data: {
      datasets: [
        {
          label: 'ECG',
          type: 'line',
          data: sigData,
          borderColor: '#3b82f6',
          borderWidth: 1.2,
          pointRadius: 0,
          tension: 0.2,
          fill: false,
          showLine: true,
          parsing: false,
        },
        {
          label: 'R-peaks',
          type: 'scatter',
          data: rpData,
          pointBackgroundColor: rpData.map(d =>
            d.beatIdx === props.highlightBeat ? '#fff'
              : (LABEL_COLORS[d.label] ?? '#64748b')
          ),
          pointBorderColor: rpData.map(d => LABEL_COLORS[d.label] ?? '#64748b'),
          pointBorderWidth: 2,
          pointRadius: rpData.map(d => d.beatIdx === props.highlightBeat ? 9 : 5),
          parsing: false,
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => `Sample ${items[0].raw.x * ds}`,
            label: (item) => {
              if (item.datasetIndex === 1) {
                const d = rpData[item.dataIndex]
                return `Beat ${d.beatIdx}: ${d.label}`
              }
              return `${item.raw.y.toFixed(4)} mV`
            },
          },
        },
      },
      scales: {
        x: { type: 'linear', ticks: { maxTicksLimit: 8, color: '#94a3b8', font: { size: 10 } }, grid: { color: '#f1f5f9' } },
        y: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: '#f1f5f9' } },
      },
      onClick: (evt, elements) => {
        const rpEls = elements.filter(e => e.datasetIndex === 1)
        if (rpEls.length) emit('select-beat', rpEls[0].index)
      },
    },
  })
}

const emit = defineEmits(['select-beat'])

onMounted(buildChart)
watch(() => [props.signal, props.rpeaks, props.beats, props.highlightBeat], buildChart, { deep: true })
onBeforeUnmount(() => chart?.destroy())
</script>
