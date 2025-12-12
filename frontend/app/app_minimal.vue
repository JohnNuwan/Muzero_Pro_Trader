<template>
  <div class="min-h-screen bg-[#0b0e14] text-white font-sans selection:bg-blue-500/30">
    <div class="fixed inset-0 pointer-events-none z-0">
      <div class="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/10 rounded-full blur-[120px]"></div>
      <div class="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-600/10 rounded-full blur-[120px]"></div>
    </div>

    <div class="relative z-10 flex flex-col h-screen overflow-hidden">
      <header class="bg-[#111827]/80 backdrop-blur-md border-b border-gray-800 p-4 flex justify-between items-center shrink-0">
        <div class="flex items-center gap-4">
          <div class="relative">
            <div class="w-3 h-3 rounded-full" :class="status.running ? 'bg-green-500 animate-pulse' : 'bg-red-500'"></div>
            <div v-if="status.running" class="absolute inset-0 bg-green-500 rounded-full animate-ping opacity-75"></div>
          </div>
          <h1 class="text-2xl font-black tracking-tight bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            GEMINI V12
          </h1>
          <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-gray-800 text-gray-400 border border-gray-700">COMMAND CENTER</span>
        </div>
        
        <div class="flex gap-8">
          <div class="text-right">
            <div class="text-[10px] uppercase tracking-wider text-gray-500 font-bold">Balance</div>
            <div class="font-mono text-xl font-bold text-white">{{ formatMoney(status.account?.balance || 0) }}</div>
          </div>
          <div class="text-right">
            <div class="text-[10px] uppercase tracking-wider text-gray-500 font-bold">Equity</div>
            <div class="font-mono text-xl font-bold text-blue-400">{{ formatMoney(status.account?.equity || 0) }}</div>
          </div>
          <div class="text-right">
            <div class="text-[10px] uppercase tracking-wider text-gray-500 font-bold">Daily PnL</div>
            <div class="font-mono text-xl font-bold" :class="status.daily_stats?.profit >= 0 ? 'text-green-400' : 'text-red-400'">
              {{ formatMoney(status.daily_stats?.profit || 0) }}
            </div>
          </div>
          <div class="text-right border-l border-gray-700 pl-6">
             <div class="text-[10px] uppercase tracking-wider text-gray-500 font-bold">Market Time</div>
             <div class="font-mono text-xl font-bold text-white">{{ currentTime }}</div>
          </div>
        </div>
      </header>

      <main class="flex-1 overflow-hidden p-6 relative">
         <div class="flex gap-4 mb-6 border-b border-gray-800 pb-1">
            <button 
               @click="activeTab = 'COMMAND'" 
               class="px-4 py-2 text-sm font-bold transition-colors relative"
               :class="activeTab === 'COMMAND' ? 'text-blue-400' : 'text-gray-500 hover:text-gray-300'">
               COMMAND CENTER
               <div v-if="activeTab === 'COMMAND'" class="absolute bottom-[-5px] left-0 right-0 h-0.5 bg-blue-400 shadow-[0_0_10px_rgba(96,165,250,0.5)]"></div>
            </button>
            <button 
               @click="activeTab = 'MARKET'" 
               class="px-4 py-2 text-sm font-bold transition-colors relative"
               :class="activeTab === 'MARKET' ? 'text-purple-400' : 'text-gray-500 hover:text-gray-300'">
               MARKET VISION
               <div v-if="activeTab === 'MARKET'" class="absolute bottom-[-5px] left-0 right-0 h-0.5 bg-purple-400 shadow-[0_0_10px_rgba(192,132,252,0.5)]"></div>
            </button>
            <button 
               @click="activeTab = 'HISTORY'" 
               class="px-4 py-2 text-sm font-bold transition-colors relative"
               :class="activeTab === 'HISTORY' ? 'text-green-400' : 'text-gray-500 hover:text-gray-300'">
               HISTORY
               <div v-if="activeTab === 'HISTORY'" class="absolute bottom-[-5px] left-0 right0 h-0.5 bg-green-400 shadow-[0_0_10px_rgba(74,222,128,0.5)]"></div>
            </button>
         </div>

         <div v-if="activeTab === 'COMMAND'" class="h-full">
            <div class="text-center py-12 text-gray-500">Command Center Content</div>
         </div>

         <div v-if="activeTab === 'MARKET'" class="h-full">
            <div class="text-center py-12 text-gray-500">Market Vision Content</div>
         </div>

         <div v-if="activeTab === 'HISTORY'" class="h-full overflow-hidden flex flex-col">
            <div class="bg-[#1f2937]/50 backdrop-blur border border-gray-700/50 rounded-xl overflow-hidden flex flex-col h-full">
               <div class="p-4 border-b border-gray-700/50 flex justify-between items-center">
                  <h2 class="text-xs font-bold tracking-wider text-slate-400 uppercase flex items-center gap-2">
                     <ClockIcon class="w-3 h-3" /> Trade History
                  </h2>
               </div>
               <div class="overflow-auto custom-scrollbar flex-1">
                  <table class="w-full text-left border-collapse">
                     <thread class="bg-gray-800/50 text-[9px] uppercase text-gray-500 font-bold tracking-wider sticky top-0 z-10 backdrop-blur">
                        <tr>
                           <th class="p-3">Time</th>
                           <th class="p-3">Ticket</th>
                           <th class="p-3">Symbol</th>
                           <th class="p-3">Type</th>
                           <th class="p-3 text-right">Volume</th>
                           <th class="p-3 text-right">Profit</th>
                        </tr>
                     </thread>
                     <tbody class="divide-y divide-gray-700/30 text-[11px] font-mono">
                        <tr v-for="t in historyData" :key="t.ticket" class="hover:bg-white/5 transition-colors">
                           <td class="p-3 text-gray-400">{{ t.time }}</td>
                           <td class="p-3 text-gray-500">#{{ t.ticket }}</td>
                           <td class="p-3 font-bold text-white">{{ t.symbol }}</td>
                           <td class="p-3">
                              <span :class="t.type === 'BUY' ? 'text-green-400' : 'text-red-400'" class="font-bold">{{ t.type }}</span>
                           </td>
                           <td class="p-3 text-right text-gray-300">{{ t.volume }}</td>
                           <td class="p-3 text-right font-bold" :class="t.profit >= 0 ? 'text-green-400' : 'text-red-400'">
                              {{ formatMoney(t.profit) }}
                           </td>
                        </tr>
                        <tr v-if="historyData.length === 0">
                           <td colspan="6" class="p-8 text-center text-gray-500 italic">No history available</td>
                        </tr>
                     </tbody>
                  </table>
               </div>
            </div>
         </div>

      </main>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ClockIcon } from 'lucide-vue-next'
import axios from 'axios'

const status = ref({ running: false, daily_stats: { profit: 0 }, account: {} })
const activeTab = ref('COMMAND')
const historyData = ref([])
const currentTime = ref(new Date().toLocaleTimeString())

const API_URL = 'http://localhost:8000/api'

const fetchStatus = async () => {
  try {
    const res = await axios.get(`${API_URL}/status`)
    status.value = res.data
  } catch (e) { console.error("API Error", e) }
}

const fetchHistory = async () => {
  try {
    const res = await axios.get(`${API_URL}/history`)
    historyData.value = res.data
  } catch (e) { console.error("API Error", e) }
}

onMounted(async () => {
  setInterval(fetchStatus, 1000)
  setInterval(fetchHistory, 5000)
  setInterval(() => {
    currentTime.value = new Date().toLocaleTimeString()
  }, 1000)
})

const formatMoney = (val) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val)
</script>

<style>
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #111827; }
::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4b5563; }
</style>
