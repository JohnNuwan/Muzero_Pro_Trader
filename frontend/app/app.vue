<template>
  <div class="min-h-screen bg-[#0b0e14] text-white font-sans">
    <div class="fixed inset-0 pointer-events-none z-0">
      <div class="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/10 rounded-full blur-[120px]"></div>
      <div class="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-600/10 rounded-full blur-[120px]"></div>
    </div>

    <div class="relative z-10 flex flex-col h-screen overflow-hidden">
      <header class="bg-[#111827]/80 backdrop-blur-md border-b border-gray-800 p-4 flex justify-between items-center shrink-0">
        <div class="flex items-center gap-4">
          <div class="w-3 h-3 rounded-full" :class="status.running ? 'bg-green-500 animate-pulse' : 'bg-red-500'"></div>
          <h1 class="text-2xl font-black bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            GEMINI V12
          </h1>
        </div>
        
        <div class="flex gap-8">
          <div class="text-right">
            <div class="text-[10px] uppercase text-gray-500 font-bold">Balance</div>
            <div class="font-mono text-xl font-bold text-white">{{ formatMoney(status.account?.balance || 0) }}</div>
          </div>
          <div class="text-right">
            <div class="text-[10px] uppercase text-gray-500 font-bold">Equity</div>
            <div class="font-mono text-xl font-bold text-blue-400">{{ formatMoney(status.account?.equity || 0) }}</div>
          </div>
          <div class="text-right">
            <div class="text-[10px] uppercase text-gray-500 font-bold">Daily PnL</div>
            <div class="font-mono text-xl font-bold" :class="status.daily_stats?.profit >= 0 ? 'text-green-400' : 'text-red-400'">
              {{ formatMoney(status.daily_stats?.profit || 0) }}
            </div>
          </div>
        </div>
      </header>

      <main class="flex-1 overflow-hidden p-6">
        <div class="flex gap-4 mb-6 border-b border-gray-800 pb-1">
          <button @click="activeTab = 'COMMAND'" class="px-4 py-2 text-sm font-bold transition-colors relative" :class="activeTab === 'COMMAND' ? 'text-blue-400' : 'text-gray-500'">
            COMMAND CENTER
            <div v-if="activeTab === 'COMMAND'" class="absolute bottom-[-5px] left-0 right-0 h-0.5 bg-blue-400"></div>
          </button>
          <button @click="activeTab = 'MARKET'" class="px-4 py-2 text-sm font-bold transition-colors relative" :class="activeTab === 'MARKET' ? 'text-purple-400' : 'text-gray-500'">
            MARKET VISION
            <div v-if="activeTab === 'MARKET'" class="absolute bottom-[-5px] left-0 right-0 h-0.5 bg-purple-400"></div>
          </button>
          <button @click="activeTab = 'HISTORY'" class="px-4 py-2 text-sm font-bold transition-colors relative" :class="activeTab === 'HISTORY' ? 'text-green-400' : 'text-gray-500'">
            HISTORY
            <div v-if="activeTab === 'HISTORY'" class="absolute bottom-[-5px] left-0 right-0 h-0.5 bg-green-400"></div>
          </button>
          <button @click="activeTab = 'STRATEGIES'" class="px-4 py-2 text-sm font-bold transition-colors relative" :class="activeTab === 'STRATEGIES' ? 'text-pink-400' : 'text-gray-500'">
            STRATEGIES
            <div v-if="activeTab === 'STRATEGIES'" class="absolute bottom-[-5px] left-0 right-0 h-0.5 bg-pink-400"></div>
          </button>
          <button @click="activeTab = 'CONFIG'" class="px-4 py-2 text-sm font-bold transition-colors relative" :class="activeTab === 'CONFIG' ? 'text-orange-400' : 'text-gray-500'">
            CONFIGURATION
            <div v-if="activeTab === 'CONFIG'" class="absolute bottom-[-5px] left-0 right-0 h-0.5 bg-orange-400"></div>
          </button>
        </div>

        <div v-if="activeTab === 'COMMAND'" class="grid grid-cols-2 gap-6 h-[calc(100vh-200px)]">
          <div class="bg-[#1f2937]/50 border border-gray-700 rounded-xl p-4 overflow-auto">
            <h2 class="text-sm font-bold mb-4 text-gray-300">Active Positions</h2>
            <div v-if="positions.length === 0" class="text-center text-gray-500 py-8">No active positions</div>
            <div v-for="pos in positions" :key="pos.ticket" class="bg-gray-800/50 border border-gray-700 p-3 rounded-lg mb-2">
              <div class="flex justify-between items-start mb-2">
                <div class="flex items-center gap-2">
                  <span class="font-bold text-white">{{ pos.symbol }}</span>
                  <span class="text-[10px] px-1.5 py-0.5 rounded font-bold" :class="pos.type === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'">{{ pos.type }}</span>
                </div>
                <span class="font-mono font-bold" :class="pos.profit >= 0 ? 'text-green-400' : 'text-red-400'">{{ formatMoney(pos.profit) }}</span>
              </div>
              <div class="flex justify-between text-xs text-gray-500">
                <span>{{ pos.volume }} Lots</span>
                <span>{{ pos.comment }}</span>
              </div>
            </div>
          </div>

          <div class="bg-[#1f2937]/50 border border-gray-700 rounded-xl p-4 overflow-auto">
            <h2 class="text-sm font-bold mb-4 text-gray-300">System Logs</h2>
            <div class="font-mono text-[10px] space-y-1">
              <div v-for="(log, i) in logs" :key="i" class="flex gap-2">
                <span class="text-gray-600">[{{ log.time }}]</span>
                <span :class="log.type === 'ERROR' ? 'text-red-400' : log.type === 'SUCCESS' ? 'text-green-400' : 'text-blue-300'">{{ log.message }}</span>
              </div>
            </div>
          </div>
        </div>

        <div v-if="activeTab === 'MARKET'" class="h-[calc(100vh-200px)]">
          <div class="bg-[#1f2937]/50 border border-gray-700 rounded-xl p-4 h-full overflow-auto">
            <h2 class="text-sm font-bold mb-4 text-gray-300">Market Matrix</h2>
            <table class="w-full text-left border-collapse">
              <thead class="bg-gray-800/50 text-[9px] uppercase text-gray-500 font-bold sticky top-0">
                <tr>
                  <th class="p-2">Symbol</th>
                  <th class="p-2">Price</th>
                  <th class="p-2">Trend</th>
                  <th class="p-2">Regime</th>
                  <th class="p-2">ADX</th>
                  <th class="p-2">RSI</th>
                  <th class="p-2">Z-Score</th>
                  <th class="p-2">Patterns</th>
                  <th class="p-2">Status</th>
                </tr>
              </thead>
              <tbody class="text-[10px] font-mono">
                <tr v-for="s in marketData" :key="s.symbol" class="hover:bg-white/5 border-b border-gray-700/30">
                  <td class="p-2 font-bold text-white">{{ s.symbol }}</td>
                  <td class="p-2 text-gray-300">{{ s.price }}</td>
                  <td class="p-2">
                    <span class="px-1.5 py-0.5 rounded text-[9px]" :class="s.trend === 'BULLISH' || s.trend === 'STRONG BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'">
                      {{ s.trend || '-' }}
                    </span>
                  </td>
                  <td class="p-2">
                    <span class="px-1.5 py-0.5 rounded text-[9px]" :class="s.regime === 'TREND' ? 'bg-purple-500/20 text-purple-400' : 'bg-blue-500/20 text-blue-400'">
                      {{ s.regime || '-' }}
                    </span>
                  </td>
                  <td class="p-2" :class="s.adx > 25 ? 'text-yellow-400 font-bold' : 'text-gray-400'">{{ s.adx || '-' }}</td>
                  <td class="p-2" :class="s.rsi > 70 ? 'text-red-400' : s.rsi < 30 ? 'text-green-400' : 'text-gray-300'">{{ s.rsi || '-' }}</td>
                  <td class="p-2 font-bold" :class="Math.abs(s.z_score) > 2 ? 'text-orange-400' : 'text-gray-300'">{{ s.z_score?.toFixed(2) || '-' }}</td>
                  <td class="p-2">
                    <div class="flex flex-wrap gap-1">
                      <span v-for="p in s.patterns" :key="p.name" 
                            class="px-1.5 py-0.5 rounded text-[8px] border"
                            :class="p.type === 'bullish' ? 'bg-green-900/30 border-green-700 text-green-400' : p.type === 'bearish' ? 'bg-red-900/30 border-red-700 text-red-400' : 'bg-gray-800 border-gray-600 text-gray-400'">
                        {{ p.name }} <span class="opacity-70 text-[7px]">({{ p.tf }})</span>
                      </span>
                      <span v-if="!s.patterns || s.patterns.length === 0" class="text-gray-600">-</span>
                    </div>
                  </td>
                  <td class="p-2">
                    <span :class="s.status === 'ACTIVE' ? 'text-green-400' : s.status === 'SLEEP' ? 'text-yellow-500' : 'text-gray-500'">{{ s.status || 'OFFLINE' }}</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div v-if="activeTab === 'HISTORY'" class="h-[calc(100vh-200px)] flex flex-col gap-4">
          <!-- Filters & Stats -->
          <div class="flex gap-4 items-center justify-between bg-[#1f2937]/50 border border-gray-700 rounded-xl p-4">
            <div class="flex gap-2">
              <button v-for="p in ['day', 'week', 'month', 'all']" :key="p" 
                      @click="historyPeriod = p; fetchHistoryAnalysis()"
                      class="px-3 py-1 rounded text-xs font-bold uppercase transition-colors"
                      :class="historyPeriod === p ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'">
                {{ p }}
              </button>
            </div>
            <div class="flex gap-6 text-right">
              <div>
                <div class="text-[10px] text-gray-500 uppercase font-bold">Total Profit</div>
                <div class="font-mono font-bold text-lg" :class="historyStats.stats?.total_profit >= 0 ? 'text-green-400' : 'text-red-400'">
                  {{ formatMoney(historyStats.stats?.total_profit || 0) }}
                </div>
              </div>
              <div>
                <div class="text-[10px] text-gray-500 uppercase font-bold">Win Rate</div>
                <div class="font-mono font-bold text-lg text-blue-400">{{ historyStats.stats?.win_rate || 0 }}%</div>
              </div>
              <div>
                <div class="text-[10px] text-gray-500 uppercase font-bold">Trades</div>
                <div class="font-mono font-bold text-lg text-white">{{ historyStats.stats?.total_trades || 0 }}</div>
              </div>
            </div>
          </div>

          <div class="flex gap-4 flex-1 overflow-hidden">
            <!-- Symbol Performance -->
            <div class="w-1/3 bg-[#1f2937]/50 border border-gray-700 rounded-xl overflow-hidden flex flex-col">
              <div class="p-3 border-b border-gray-700 bg-gray-800/30">
                <h3 class="text-xs font-bold text-gray-300 uppercase">Performance by Symbol</h3>
              </div>
              <div class="overflow-auto flex-1 p-2 space-y-2">
                <div v-for="s in historyStats.by_symbol" :key="s.symbol" class="bg-gray-800/40 p-2 rounded flex justify-between items-center">
                  <div>
                    <div class="font-bold text-xs text-white">{{ s.symbol }}</div>
                    <div class="text-[10px] text-gray-500">{{ s.wins }}/{{ s.total }} Wins</div>
                  </div>
                  <div class="text-right">
                    <div class="font-mono text-xs font-bold" :class="s.profit >= 0 ? 'text-green-400' : 'text-red-400'">
                      {{ formatMoney(s.profit) }}
                    </div>
                    <div class="text-[9px] text-gray-500">{{ Math.round((s.wins/s.total)*100) }}% WR</div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Trade List -->
            <div class="w-2/3 bg-[#1f2937]/50 border border-gray-700 rounded-xl overflow-hidden flex flex-col">
              <div class="p-3 border-b border-gray-700 bg-gray-800/30">
                <h3 class="text-xs font-bold text-gray-300 uppercase">Trade Log</h3>
              </div>
              <div class="overflow-auto flex-1">
                <table class="w-full text-left border-collapse">
                  <thead class="bg-gray-800/50 text-[9px] uppercase text-gray-500 font-bold sticky top-0">
                    <tr>
                      <th class="p-2">Time</th>
                      <th class="p-2">Symbol</th>
                      <th class="p-2">Type</th>
                      <th class="p-2 text-right">Volume</th>
                      <th class="p-2 text-right">Profit</th>
                    </tr>
                  </thead>
                  <tbody class="text-[10px] font-mono">
                    <tr v-for="t in historyStats.trades" :key="t.ticket" class="hover:bg-white/5 border-b border-gray-700/30">
                      <td class="p-2 text-gray-400">{{ t.time }}</td>
                      <td class="p-2 font-bold text-white">{{ t.symbol }}</td>
                      <td class="p-2">
                        <span :class="t.type === 'BUY' ? 'text-green-400' : 'text-red-400'" class="font-bold">{{ t.type }}</span>
                      </td>
                      <td class="p-2 text-right text-gray-300">{{ t.volume }}</td>
                      <td class="p-2 text-right font-bold" :class="t.profit >= 0 ? 'text-green-400' : 'text-red-400'">
                        {{ formatMoney(t.profit) }}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

      </main>

        <div v-if="activeTab === 'STRATEGIES'" class="h-[calc(100vh-200px)] p-6">
          <Strategies :strategies="strategiesData" :models="marketData" />
        </div>

        <div v-if="activeTab === 'CONFIG'" class="h-[calc(100vh-200px)] overflow-auto">
          <div class="flex justify-end mb-4">
            <button @click="saveConfig" class="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded font-bold text-xs transition-colors flex items-center gap-2">
              <span>💾</span> SAVE CONFIGURATION
            </button>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <!-- Global Settings -->
            <div class="bg-[#1f2937]/50 border border-gray-700 rounded-xl p-6">
              <h2 class="text-lg font-bold text-orange-400 mb-4 flex items-center gap-2">
                <span>⚙️</span> Global Settings
              </h2>
              <div class="space-y-4">
                <div v-for="(val, key) in configData.global" :key="key" class="flex justify-between items-center border-b border-gray-700/50 pb-2">
                  <span class="text-gray-400 capitalize text-xs">{{ key.replace(/_/g, ' ') }}</span>
                  <input v-model="configData.global[key]" type="text" class="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-right text-white font-mono text-xs w-24 focus:border-blue-500 outline-none">
                </div>
              </div>
            </div>

            <!-- Symbol Settings -->
            <div class="bg-[#1f2937]/50 border border-gray-700 rounded-xl p-6 col-span-1 md:col-span-2">
              <h2 class="text-lg font-bold text-blue-400 mb-4 flex items-center gap-2">
                <span>📊</span> Symbol Configuration
              </h2>
              <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div v-for="(params, sym) in configData.symbols" :key="sym" class="bg-gray-800/40 rounded-lg p-4 border border-gray-700 hover:border-blue-500/50 transition-colors">
                  <div class="flex justify-between items-center mb-3 border-b border-gray-700 pb-2">
                    <span class="font-bold text-white">{{ sym }}</span>
                    <span class="text-[10px] px-2 py-0.5 bg-blue-500/20 text-blue-300 rounded">ACTIVE</span>
                  </div>
                  <div class="space-y-2 text-xs">
                    <div v-for="(val, key) in params" :key="key" class="flex justify-between items-center">
                      <span class="text-gray-500">{{ key }}</span>
                      <input v-model="configData.symbols[sym][key]" type="text" class="bg-gray-900 border border-gray-700 rounded px-2 py-0.5 text-right text-gray-300 font-mono w-20 focus:border-blue-500 outline-none">
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import Strategies from '../components/Strategies.vue'

const status = ref({ running: false, daily_stats: { profit: 0 }, account: {} })
const activeTab = ref('COMMAND')
const positions = ref([])
const logs = ref([])
const marketData = ref([])
const historyData = ref([]) // Kept for compatibility if needed, but mainly using historyStats
const historyStats = ref({ stats: {}, trades: [], by_symbol: [] })
const historyPeriod = ref('all')
const configData = ref({ global: {}, symbols: {} })
const strategiesData = ref([])

const API_URL = 'http://localhost:8000/api'

const fetchStatus = async () => {
  try {
    const res = await axios.get(`${API_URL}/status`)
    status.value = res.data
  } catch (e) { console.error("API Error", e) }
}

const fetchPositions = async () => {
  try {
    const res = await axios.get(`${API_URL}/positions`)
    positions.value = res.data
  } catch (e) { console.error("API Error", e) }
}

const fetchLogs = async () => {
  try {
    const res = await axios.get(`${API_URL}/logs`)
    logs.value = res.data
  } catch (e) { console.error("API Error", e) }
}

const fetchMarket = async () => {
  try {
    const res = await axios.get(`${API_URL}/market`)
    marketData.value = Object.values(res.data)
  } catch (e) { console.error("API Error", e) }
}

const fetchHistoryAnalysis = async () => {
  try {
    const res = await axios.get(`${API_URL}/history/analysis?period=${historyPeriod.value}`)
    historyStats.value = res.data
  } catch (e) { console.error("API Error", e) }
}

const fetchConfig = async () => {
  try {
    const res = await axios.get(`${API_URL}/config`)
    configData.value = res.data
  } catch (e) { console.error("API Error", e) }
}

const fetchStrategies = async () => {
  try {
    const res = await axios.get(`${API_URL}/strategies`)
    strategiesData.value = res.data
  } catch (e) { console.error("API Error", e) }
}

const saveConfig = async () => {
  try {
    await axios.post(`${API_URL}/config/update`, configData.value)
    alert("Configuration saved successfully!")
  } catch (e) { 
    console.error("API Error", e)
    alert("Failed to save configuration")
  }
}

onMounted(() => {
  setInterval(fetchStatus, 1000)
  setInterval(fetchPositions, 1000)
  setInterval(fetchLogs, 1000)
  setInterval(fetchMarket, 1000)
  
  // Initial fetches
  fetchHistoryAnalysis()
  fetchConfig()
  fetchStrategies()
  setInterval(fetchStrategies, 5000) // Refresh strategies every 5s
})

const formatMoney = (val) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val)
</script>

<style>
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #111827; }
::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
</style>
