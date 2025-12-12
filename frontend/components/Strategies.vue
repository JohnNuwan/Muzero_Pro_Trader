<template>
  <div class="h-full flex flex-col gap-6">
    <!-- Strategy Performance -->
    <div class="bg-[#1f2937]/50 border border-gray-700 rounded-xl p-6 flex-1 overflow-hidden flex flex-col">
      <h2 class="text-lg font-bold text-purple-400 mb-4 flex items-center gap-2">
        <span>♟️</span> Strategy Performance
      </h2>
      
      <div class="overflow-auto flex-1">
        <table class="w-full text-left border-collapse">
          <thead class="bg-gray-800/50 text-[10px] uppercase text-gray-500 font-bold sticky top-0">
            <tr>
              <th class="p-3">Strategy Name</th>
              <th class="p-3 text-right">Total Trades</th>
              <th class="p-3 text-right">Win Rate</th>
              <th class="p-3 text-right">Net Profit</th>
            </tr>
          </thead>
          <tbody class="text-xs font-mono">
            <tr v-if="strategies.length === 0">
              <td colspan="4" class="p-4 text-center text-gray-500">No strategy data available yet.</td>
            </tr>
            <tr v-for="strat in strategies" :key="strat.name" class="hover:bg-white/5 border-b border-gray-700/30 transition-colors">
              <td class="p-3 font-bold text-white flex items-center gap-2">
                <span class="w-2 h-2 rounded-full" :class="getStrategyColor(strat.name)"></span>
                {{ strat.name }}
              </td>
              <td class="p-3 text-right text-gray-300">{{ strat.trades }}</td>
              <td class="p-3 text-right">
                <div class="flex items-center justify-end gap-2">
                  <div class="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                    <div class="h-full bg-blue-500" :style="{ width: `${strat.win_rate}%` }"></div>
                  </div>
                  <span :class="strat.win_rate > 50 ? 'text-blue-400' : 'text-gray-400'">{{ strat.win_rate }}%</span>
                </div>
              </td>
              <td class="p-3 text-right font-bold" :class="strat.profit >= 0 ? 'text-green-400' : 'text-red-400'">
                {{ formatMoney(strat.profit) }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- AI Model Status -->
    <div class="bg-[#1f2937]/50 border border-gray-700 rounded-xl p-6 h-1/3 overflow-hidden flex flex-col">
      <h2 class="text-lg font-bold text-blue-400 mb-4 flex items-center gap-2">
        <span>🧠</span> AI Model Status
      </h2>
      
      <div class="overflow-auto flex-1">
        <table class="w-full text-left border-collapse">
          <thead class="bg-gray-800/50 text-[10px] uppercase text-gray-500 font-bold sticky top-0">
            <tr>
              <th class="p-3">Symbol</th>
              <th class="p-3">Status</th>
              <th class="p-3">Last Trained</th>
              <th class="p-3 text-right">Confidence</th>
              <th class="p-3 text-right">Prediction</th>
            </tr>
          </thead>
          <tbody class="text-xs font-mono">
            <tr v-for="model in models" :key="model.symbol" class="hover:bg-white/5 border-b border-gray-700/30 transition-colors">
              <td class="p-3 font-bold text-white">{{ model.symbol }}</td>
              <td class="p-3">
                <span class="px-2 py-0.5 rounded text-[10px] font-bold" 
                      :class="model.evolving ? 'bg-yellow-500/20 text-yellow-400 animate-pulse' : 'bg-green-500/20 text-green-400'">
                  {{ model.evolving ? 'EVOLVING...' : 'ACTIVE' }}
                </span>
              </td>
              <td class="p-3 text-gray-400">{{ model.last_trained || 'Never' }}</td>
              <td class="p-3 text-right font-bold" :class="model.confidence > 80 ? 'text-green-400' : 'text-gray-400'">
                {{ model.confidence || 0 }}%
              </td>
              <td class="p-3 text-right">
                <span v-if="model.prediction" class="px-2 py-0.5 rounded text-[10px] font-bold"
                      :class="model.prediction === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'">
                  {{ model.prediction }}
                </span>
                <span v-else class="text-gray-600">-</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup>
import { defineProps } from 'vue'

const props = defineProps({
  strategies: { type: Array, default: () => [] },
  models: { type: Array, default: () => [] }
})

const formatMoney = (val) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val)

const getStrategyColor = (name) => {
  switch (name) {
    case 'TREND': return 'bg-purple-500'
    case 'REVERSION': return 'bg-blue-500'
    case 'SNIPER': return 'bg-green-500'
    default: return 'bg-gray-500'
  }
}
</script>
