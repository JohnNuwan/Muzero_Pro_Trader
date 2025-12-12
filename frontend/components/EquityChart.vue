<template>
  <div ref="chartContainer" class="w-full h-full rounded-lg overflow-hidden"></div>
</template>

<script setup>
import { ref, onMounted, watch, onUnmounted } from 'vue'
import { createChart } from 'lightweight-charts'

const props = defineProps({
  data: {
    type: Array,
    default: () => []
  }
})

const chartContainer = ref(null)
let chart = null
let series = null

const initChart = () => {
  if (!chartContainer.value) return

  chart = createChart(chartContainer.value, {
    layout: { background: { color: 'transparent' }, textColor: '#6b7280' },
    grid: { vertLines: { color: '#374151', style: 2 }, horzLines: { color: '#374151', style: 2 } },
    width: chartContainer.value.clientWidth,
    height: chartContainer.value.clientHeight,
    rightPriceScale: { borderVisible: false },
    timeScale: { borderVisible: false, timeVisible: false },
    handleScale: false,
    handleScroll: false,
  })

  series = chart.addAreaSeries({
    lineColor: '#60a5fa',
    topColor: 'rgba(96, 165, 250, 0.4)',
    bottomColor: 'rgba(96, 165, 250, 0.0)',
    lineWidth: 2
  })

  if (props.data.length > 0) {
    series.setData(props.data)
    chart.timeScale().fitContent()
  }

  window.addEventListener('resize', handleResize)
}

const handleResize = () => {
  if (chart && chartContainer.value) {
    chart.applyOptions({ width: chartContainer.value.clientWidth, height: chartContainer.value.clientHeight })
  }
}

watch(() => props.data, (newData) => {
  if (series && newData.length > 0) {
    series.setData(newData)
    chart.timeScale().fitContent()
  }
}, { deep: true })

onMounted(() => {
  initChart()
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  if (chart) {
    chart.remove()
  }
})
</script>
