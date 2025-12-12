<template>
  <div class="min-h-screen bg-[#0b0e14] text-white font-sans selection:bg-blue-500/30">
    <!-- Background Gradient Mesh -->
    <div class="fixed inset-0 pointer-events-none z-0">
      <div class="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/10 rounded-full blur-[120px]"></div>
      <div class="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-600/10 rounded-full blur-[120px]"></div>
    </div>

    <div class="relative z-10 flex flex-col h-screen overflow-hidden">
      <!-- Header -->
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

      <!-- Main Content -->
      <main class="flex-1 overflow-hidden p-6 relative">
         <!-- Tabs -->
         <div class="flex gap-4 mb-6 border-b border-gray-800 pb-1">
            <button 
               @click="activeTab = 'COMMAND'" 
               class="px-4 py-2 text-sm font-bold transition-colors relative"
               :class="activeTab === 'COMMAND' ? 'text-blue-400' : 'text-gray-500 hover:text-gray-300'"
            >
               COMMAND CENTER
               <div v-if="activeTab === 'COMMAND'" class="absolute bottom-[-5px] left-0 right-0 h-0.5 bg-blue-400 shadow-[0_0_10px_rgba(96,165,250,0.5)]"></div>
            </button>
            <button 
               @click="activeTab = 'MARKET'" 
               class="px-4 py-2 text-sm font-bold transition-colors relative"
               :class="activeTab === 'MARKET' ? 'text-purple-400' : 'text-gray-500 hover:text-gray-300'"
            >
               MARKET VISION
               <div v-if="activeTab === 'MARKET'" class="absolute bottom-[-5px] left-0 right-0 h-0.5 bg-purple-400 shadow-[0_0_10px_rgba(192,132,252,0.5)]"></div>
            </button>
            <button 
               @click="activeTab = 'HISTORY'" 
               class="px-4 py-2 text-sm font-bold transition-colors relative"
               :class="activeTab === 'HISTORY' ? 'text-green-400' : 'text-gray-500 hover:text-gray-300'"
            >
               HISTORY
               <div v-if="activeTab === 'HISTORY'" class="absolute bottom-[-5px] left-0 right-0 h-0.5 bg-green-400 shadow-[0_0_10px_rgba(74,222,128,0.5)]"></div>
            </button>
         </div>


         <!-- TAB 1: COMMAND CENTER -->
         <div v-if="activeTab === 'COMMAND'" class="grid grid-cols-12 gap-6 h-full overflow-hidden">
             <!-- Left: KPI & Positions -->
             <div class="col-span-8 flex flex-col gap-6 h-full overflow-hidden">
                <!-- KPI Cards -->
                <div class="grid grid-cols-4 gap-4 shrink-0">
                   <div class="bg-[#1f2937]/50 backdrop-blur border border-gray-700/50 p-4 rounded-xl flex flex-col justify-between group hover:border-blue-500/30 transition">
                      <div class="text-gray-400 text-xs font-bold uppercase tracking-wider flex items-center gap-2">
                      <ActivityIcon class="w-3 h-3" /> Sharpe Ratio
                      </div>
                      <div class="text-2xl font-mono font-bold text-white group-hover:text-blue-400 transition">{{ advStats.sharpe || '0.00' }}</div>
                   </div>
                   <div class="bg-[#1f2937]/50 backdrop-blur border border-gray-700/50 p-4 rounded-xl flex flex-col justify-between group hover:border-red-500/30 transition">
                      <div class="text-gray-400 text-xs font-bold uppercase tracking-wider flex items-center gap-2">
                      <TrendingDownIcon class="w-3 h-3" /> Max Drawdown
                      </div>
                      <div class="text-2xl font-mono font-bold text-red-400">{{ formatMoney(advStats.drawdown || 0) }}</div>
                   </div>
                   <div class="bg-[#1f2937]/50 backdrop-blur border border-gray-700/50 p-4 rounded-xl flex flex-col justify-between group hover:border-green-500/30 transition">
                      <div class="text-gray-400 text-xs font-bold uppercase tracking-wider flex items-center gap-2">
                      <BarChart2Icon class="w-3 h-3" /> Profit Factor
                      </div>
                      <div class="text-2xl font-mono font-bold text-green-400">{{ advStats.profit_factor || '0.00' }}</div>
                   </div>
                   <div class="bg-[#1f2937]/50 backdrop-blur border border-gray-700/50 p-4 rounded-xl flex flex-col justify-between group hover:border-purple-500/30 transition">
                      <div class="text-gray-400 text-xs font-bold uppercase tracking-wider flex items-center gap-2">
                      <TargetIcon class="w-3 h-3" /> Win Rate
                      </div>
                      <div class="text-2xl font-mono font-bold text-purple-400">{{ advStats.win_rate || '0.0' }}%</div>
                   </div>
                </div>

                <!-- Active Positions -->
                <div class="bg-[#1f2937]/50 backdrop-blur border border-gray-700/50 rounded-xl overflow-hidden flex flex-col flex-1 min-h-[200px]">
                   <div class="p-3 border-b border-gray-700/50 flex justify-between items-center bg-gray-800/30">
                   <h2 class="font-bold text-sm text-gray-300 flex items-center gap-2">
                      <BriefcaseIcon class="w-4 h-4 text-green-400" /> Active Positions
                   </h2>
                   <span class="text-xs font-mono text-gray-500">{{ positions.length }} Open</span>
                   </div>
                   <div class="overflow-y-auto flex-1 p-2 space-y-2">
                   <div v-if="positions.length === 0" class="h-full flex flex-col items-center justify-center text-gray-600 gap-2">
                      <BriefcaseIcon class="w-8 h-8 opacity-20" />
                      <span class="text-xs">No active positions</span>
                   </div>
                   <div v-for="pos in positions" :key="pos.ticket" class="bg-gray-800/50 border border-gray-700/50 p-3 rounded-lg hover:bg-gray-700/50 transition group">
                      <div class="flex justify-between items-start mb-2">
                         <div class="flex items-center gap-2">
                         <span class="font-bold text-white">{{ pos.symbol }}</span>
                         <span class="text-[10px] px-1.5 py-0.5 rounded font-bold" :class="pos.type === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'">{{ pos.type }}</span>
                         </div>
                         <span class="font-mono font-bold" :class="pos.profit >= 0 ? 'text-green-400' : 'text-red-400'">{{ formatMoney(pos.profit) }}</span>
                      </div>
                      <div class="flex justify-between items-center text-xs text-gray-500">
                         <span class="font-mono">{{ pos.volume }} Lots</span>
                         <span class="truncate max-w-[100px]">{{ pos.comment }}</span>
                      </div>
                   </div>
                   </div>
                </div>
             </div>

             <!-- Right: Controls & Logs -->
             <div class="col-span-4 flex flex-col gap-6 h-full overflow-hidden">
                <!-- Controls -->
                <div class="grid grid-cols-2 gap-3 shrink-0">
                   <button @click="toggleBot" :class="status.running ? 'bg-red-500 hover:bg-red-600 text-white shadow-lg shadow-red-500/20' : 'bg-green-500 hover:bg-green-600 text-white shadow-lg shadow-green-500/20'" class="p-4 rounded-xl font-bold transition flex items-center justify-center gap-2">
                      <PowerIcon class="w-5 h-5" />
                      {{ status.running ? 'STOP SYSTEM' : 'START SYSTEM' }}
                   </button>
                   <button @click="openConfig" class="bg-gray-700 hover:bg-gray-600 text-white p-4 rounded-xl font-bold transition flex items-center justify-center gap-2 border border-gray-600">
                      <SettingsIcon class="w-5 h-5" />
                      CONFIG
                   </button>
                </div>

                <!-- Logs -->
                <div class="bg-[#1f2937]/50 backdrop-blur border border-gray-700/50 rounded-xl overflow-hidden flex flex-col flex-1 shrink-0">
                  <div class="p-3 border-b border-gray-700/50 flex justify-between items-center bg-gray-800/30">
                  <h2 class="font-bold text-sm text-gray-300 flex items-center gap-2">
                     <TerminalIcon class="w-4 h-4 text-orange-400" /> System Logs
                  </h2>
                  </div>
                  <div class="flex-1 overflow-y-auto p-3 font-mono text-[10px] space-y-1.5 bg-black/20">
                  <div v-for="(log, i) in logs" :key="i" class="flex gap-2 opacity-80 hover:opacity-100 transition">
                     <span class="text-gray-600">[{{ log.time }}]</span>
                     <span :class="getLogColor(log.type)">{{ log.message }}</span>
                  </div>
                  </div>
               </div>
            </div>
        </div>

        <!-- TAB 2: MARKET VISION -->
        <div v-if="activeTab === 'MARKET'" class="grid grid-cols-12 gap-6 h-full overflow-hidden">
            <!-- LIVE ALGORITHM VISION -->
            <div class="col-span-12 lg:col-span-8 glass-panel p-6 relative overflow-hidden flex flex-col">
               <div class="flex justify-between items-center mb-4 shrink-0">
                  <h2 class="text-lg font-bold tracking-wider text-cyan-400 flex items-center gap-2">
                  <ActivityIcon class="w-5 h-5" />
                  LIVE ALGORITHM VISION
                  </h2>
                  <div class="flex gap-2">
                     <span class="text-xs px-2 py-1 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">V9 LOGIC ACTIVE</span>
                  </div>
               </div>

               <div class="grid grid-cols-1 md:grid-cols-2 gap-4 overflow-y-auto custom-scrollbar pr-2 flex-1">
                  <div v-for="m in marketData" :key="m.symbol" class="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50 hover:border-cyan-500/30 transition-all h-fit">
                     <div class="flex justify-between items-center mb-2">
                        <span class="font-bold text-white">{{ m.symbol }}</span>
                        <span :class="m.status === 'ACTIVE' ? 'text-green-400' : 'text-yellow-400'" class="text-xs font-mono px-2 py-0.5 rounded bg-black/30">
                           {{ m.status }}
                        </span>
                     </div>
                     
                     <div class="grid grid-cols-2 gap-2 text-xs mb-3">
                        <div class="flex flex-col">
                           <span class="text-slate-400">Regime</span>
                           <span :class="m.regime === 'TREND' ? 'text-purple-400' : 'text-orange-400'" class="font-bold">{{ m.regime || 'WAIT' }}</span>
                        </div>
                        <div class="flex flex-col text-right">
                           <span class="text-slate-400">Confidence</span>
                           <span class="font-bold text-white">{{ m.confidence || 0 }}%</span>
                        </div>
                     </div>

                     <!-- Session Badge -->
                     <div class="mb-3">
                        <span class="text-[9px] px-1.5 py-0.5 rounded bg-gray-700 text-gray-300 border border-gray-600">{{ m.session || 'OVERNIGHT' }}</span>
                     </div>

                     <!-- Indicators Bar -->
                     <div class="flex flex-col bg-black/20 rounded p-2 mb-2">
                        <div class="grid grid-cols-4 gap-2 mb-4">
            <div class="text-center">
              <div class="text-[10px] text-gray-500 font-bold">RSI</div>
              <div class="font-mono text-sm font-bold" :class="m.rsi > 70 ? 'text-red-400' : (m.rsi < 30 ? 'text-green-400' : 'text-white')">
                {{ m.rsi?.toFixed(1) || '-' }}
              </div>
            </div>
            <div class="text-center">
              <div class="text-[10px] text-gray-500 font-bold">MFI</div>
              <div class="font-mono text-sm font-bold text-blue-300">{{ m.mfi?.toFixed(1) || '-' }}</div>
            </div>
            <div class="text-center">
              <div class="text-[10px] text-gray-500 font-bold">ADX</div>
              <div class="font-mono text-sm font-bold text-yellow-300">{{ m.adx?.toFixed(1) || '-' }}</div>
            </div>
            <div class="text-center">
              <div class="text-[10px] text-gray-500 font-bold">Z-SCORE</div>
              <div class="font-mono text-sm font-bold" :class="Math.abs(m.z_score) > 2 ? 'text-purple-400' : 'text-gray-400'">
                {{ m.z_score?.toFixed(2) || '-' }}
              </div>
            </div>
          </div>

          <!-- Key Levels -->
          <div class="grid grid-cols-2 gap-x-4 gap-y-2 mb-4 bg-black/20 p-2 rounded border border-gray-700/30">
            <div class="flex justify-between text-xs">
              <span class="text-gray-500">Pivot</span>
              <span class="font-mono text-gray-300">{{ m.pivots?.Pivot?.toFixed(2) || '-' }}</span>
            </div>
            <div class="flex justify-between text-xs">
              <span class="text-gray-500">Fibo 61.8%</span>
              <span class="font-mono text-gray-300">{{ m.fibonacci?.['0.618']?.toFixed(2) || '-' }}</span>
            </div>
            <div class="flex justify-between text-xs">
              <span class="text-gray-500">R1</span>
              <span class="font-mono text-red-300/70">{{ m.pivots?.R1?.toFixed(2) || '-' }}</span>
            </div>
            <div class="flex justify-between text-xs">
              <span class="text-gray-500">S1</span>
              <span class="font-mono text-green-300/70">{{ m.pivots?.S1?.toFixed(2) || '-' }}</span>
            </div>
          </div>
                        <div v-if="m.prediction" class="mt-auto p-2 rounded text-center font-bold text-xs border transition-colors"
                           :class="m.prediction === 'BUY' ? 'bg-green-500/10 text-green-400 border-green-500/30' : 'bg-red-500/10 text-red-400 border-red-500/30'">
                           {{ m.prediction }} SIGNAL
                        </div>
                     </div>
                  </div>
               </div>
            </div>

            <!-- Market Matrix -->
            <div class="col-span-12 lg:col-span-4 bg-[#1f2937]/50 backdrop-blur border border-gray-700/50 rounded-xl overflow-hidden flex flex-col h-full">
               <div class="p-4 border-b border-gray-700/50 flex justify-between items-center">
                  <h2 class="text-xs font-bold tracking-wider text-slate-400 uppercase flex items-center gap-2">
                     <GridIcon class="w-3 h-3" /> Market Matrix
                  </h2>
                  <button class="text-[10px] bg-blue-600 hover:bg-blue-500 text-white px-2 py-1 rounded transition">LIVE SCAN</button>
               </div>
               <div class="overflow-auto custom-scrollbar flex-1">
                  <table class="w-full text-left border-collapse">
                     <thead class="bg-gray-800/50 text-[9px] uppercase text-gray-500 font-bold tracking-wider sticky top-0 z-10 backdrop-blur">
                        <tr>
                           <th class="p-2">Symbol</th>
                           <th class="p-2">Price</th>
                           <th class="p-2">Trend</th>
                           <th class="p-2">Z-Score</th>
                           <th class="p-2">Patterns</th>
                           <th class="p-2">Status</th>
                           <th class="p-2 text-right">Action</th>
                        </tr>
                     </thead>
                     <tbody class="divide-y divide-gray-700/30 text-[10px] font-mono">
                        <tr v-for="s in marketData" :key="s.symbol" class="hover:bg-white/5 transition-colors group">
                           <td class="p-2 font-bold text-white">
                              {{ s.symbol }}
                              <span v-if="s.pyramid_count > 0" class="ml-2 text-[9px] bg-blue-500/20 text-blue-400 px-1 rounded border border-blue-500/30">
                                 x{{ s.pyramid_count }}
                              </span>
                           </td>
                           <td class="p-2 text-gray-300">{{ s.price }}</td>
                           <td class="p-2">
                              <span class="px-1.5 py-0.5 rounded border text-[9px] font-bold" :class="getTrendColor(s.trend_h1)">
                                 {{ s.trend_h1 }}
                              </span>
                           </td>
                           <td class="p-2 font-bold" :class="getZScoreColor(s.z_score)">{{ s.z_score?.toFixed(2) }}</td>
                           <td class="p-2">
                              <div class="flex flex-wrap gap-1">
                                 <span v-for="p in s.patterns" :key="p" class="px-1 rounded bg-purple-500/20 text-purple-300 border border-purple-500/30 text-[8px]">
                                    {{ p }}
                                 </span>
                                 <span v-if="!s.patterns || s.patterns.length === 0" class="text-gray-600">-</span>
                              </div>
                           </td>
                           <td class="p-2">
                              <span class="flex items-center gap-1.5">
                                 <span class="w-1.5 h-1.5 rounded-full" :class="s.status === 'ACTIVE' ? 'bg-green-500 shadow-[0_0_5px_rgba(34,197,94,0.5)]' : 'bg-gray-500'"></span>
                                 <span :class="s.status === 'ACTIVE' ? 'text-green-400' : 'text-gray-500'">{{ s.status || 'OFFLINE' }}</span>
                              </span>
                           </td>
                           <td class="p-2 text-right">
                              <button @click="evolve(s.symbol)" :disabled="s.evolving" 
                                 class="px-2 py-1 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30 hover:bg-purple-500/40 disabled:opacity-50 disabled:cursor-not-allowed transition text-[9px]">
                                 {{ s.evolving ? 'EVOLVING...' : 'EVOLVE' }}
                              </button>
                           </td>
                        </tr>
                     </tbody>
                  </table>
               </div>
            </div>
            </div>
        </div>

         <!-- TAB 3: HISTORY -->
         <div v-if="activeTab === 'HISTORY'" class="h-full overflow-hidden flex flex-col">
            <div class="bg-[#1f2937]/50 backdrop-blur border border-gray-700/50 rounded-xl overflow-hidden flex flex-col h-full">
               <div class="p-4 border-b border-gray-700/50 flex justify-between items-center">
                  <h2 class="text-xs font-bold tracking-wider text-slate-400 uppercase flex items-center gap-2">
                     <ClockIcon class="w-3 h-3" /> Trade History
                  </h2>
               </div>
               <div class="overflow-auto custom-scrollbar flex-1">
                  <table class="w-full text-left border-collapse">
                     <thead class="bg-gray-800/50 text-[9px] uppercase text-gray-500 font-bold tracking-wider sticky top-0 z-10 backdrop-blur">
                        <tr>
                           <th class="p-3">Time</th>
                           <th class="p-3">Ticket</th>
                           <th class="p-3">Symbol</th>
                           <th class="p-3">Type</th>
                           <th class="p-3 text-right">Volume</th>
                           <th class="p-3 text-right">Profit</th>
                        </tr>
                     </thead>
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

    <!-- Modals (Config & Inspector) -->
    <!-- Config Modal -->
    <div v-if="showConfig" class="fixed inset-0 bg-black/90 backdrop-blur-sm flex items-center justify-center p-4 z-50" @click.self="showConfig = false">
      <div class="bg-gray-800 rounded-2xl border border-gray-700 p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto shadow-2xl shadow-black/50">
        <div class="flex justify-between items-center mb-6">
          <h3 class="text-xl font-bold flex items-center gap-2 text-white">
            <SettingsIcon class="w-5 h-5 text-blue-400" /> System Configuration
          </h3>
          <button @click="showConfig = false" class="text-gray-400 hover:text-white transition">✕</button>
        </div>

        <div v-if="configData" class="space-y-6">
          <div v-for="(params, sym) in configData" :key="sym" class="bg-gray-900/50 p-4 rounded-xl border border-gray-700/50">
            <h4 class="font-bold text-blue-300 mb-3 flex items-center gap-2">
              <div class="w-2 h-2 rounded-full bg-blue-500"></div> {{ sym }}
            </h4>
            <div class="grid grid-cols-2 gap-4">
              <div v-for="(val, key) in params" :key="key" class="flex flex-col">
                <label class="text-[10px] uppercase font-bold text-gray-500 mb-1">{{ key }}</label>
                <input v-if="typeof val !== 'object'" v-model="configData[sym][key]" :type="typeof val === 'number' ? 'number' : 'text'" step="any" class="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-white focus:border-blue-500 outline-none focus:ring-1 focus:ring-blue-500 transition" />
                <div v-else class="text-xs text-gray-400 italic p-2 bg-gray-800 rounded border border-gray-700">Complex Object (Edit in JSON)</div>
              </div>
            </div>
          </div>
          
          <div class="flex justify-end pt-4 border-t border-gray-700">
            <button @click="saveConfig" class="bg-blue-600 hover:bg-blue-500 text-white px-6 py-2 rounded-lg font-bold transition shadow-lg shadow-blue-600/20">
              Save Changes
            </button>
          </div>
        </div>
        <div v-else class="text-center py-12 text-gray-500 animate-pulse">
          Loading configuration...
        </div>
      </div>
    </div>

    <!-- Model Inspector Modal -->
    <div v-if="selectedSymbol" class="fixed inset-0 bg-black/90 backdrop-blur-sm flex items-center justify-center p-4 z-50" @click.self="selectedSymbol = null">
      <div class="bg-gray-800 rounded-2xl border border-gray-700 p-6 max-w-5xl w-full h-[85vh] flex flex-col shadow-2xl shadow-black/50">
        <div class="flex justify-between items-center mb-6">
          <h3 class="text-xl font-bold flex items-center gap-2 text-white">
            <BrainIcon class="w-5 h-5 text-purple-400" /> {{ selectedSymbol }} Analysis
          </h3>
          <button @click="selectedSymbol = null" class="text-gray-400 hover:text-white transition">✕</button>
        </div>
        
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 overflow-hidden">
          <!-- Left: Stats -->
          <div class="lg:col-span-1 space-y-4 overflow-y-auto pr-2">
             <div v-if="modelDetails" class="space-y-6">
              <div class="grid grid-cols-2 gap-4">
                <div class="bg-gray-700/30 p-4 rounded-xl border border-gray-700/50 text-center">
                  <div class="text-[10px] uppercase font-bold text-gray-500 mb-1">Prediction</div>
                  <div class="text-2xl font-black" :class="modelDetails.prediction === 1 ? 'text-green-400' : 'text-red-400'">
                    {{ modelDetails.prediction === 1 ? 'BUY' : 'SELL' }}
                  </div>
                </div>
                <div class="bg-gray-700/30 p-4 rounded-xl border border-gray-700/50 text-center">
                  <div class="text-[10px] uppercase font-bold text-gray-500 mb-1">Confidence</div>
                  <div class="text-2xl font-black text-blue-400">{{ (modelDetails.confidence * 100).toFixed(1) }}%</div>
                </div>
              </div>

              <div>
                <h4 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3">Feature Vector</h4>
                <div class="space-y-2">
                  <div v-for="(val, key) in modelDetails.features" :key="key" class="bg-gray-700/20 p-3 rounded-lg flex justify-between items-center border border-gray-700/30">
                    <span class="text-sm text-gray-400 font-medium">{{ key }}</span>
                    <span class="font-mono text-white font-bold">{{ val.toFixed(2) }}</span>
                  </div>
                </div>
              </div>
              
              <div class="text-[10px] text-gray-600 text-center mt-4">
                Last Update: {{ new Date(modelDetails.last_update).toLocaleTimeString() }}
              </div>
            </div>
            <div v-else class="text-center py-12 text-gray-500 animate-pulse">
              Analyzing market data...
            </div>
          </div>

          <!-- Right: Chart -->
          <div class="lg:col-span-2 bg-black/40 rounded-xl border border-gray-700/50 p-4 flex flex-col relative">
            <div class="absolute top-4 left-4 z-10 text-xs font-bold text-gray-500 uppercase tracking-wider">Price Action (M15)</div>
            <div ref="chartContainer" class="flex-1 w-full h-full min-h-[300px] rounded-lg overflow-hidden"></div>
          </div>
        </div>
      </div>
    </div>

  </div>
</template>

<script setup>
import { ref, onMounted, watch, nextTick } from 'vue'
import { ActivityIcon, BriefcaseIcon, CpuIcon, PowerIcon, SettingsIcon, TerminalIcon, BrainIcon, HistoryIcon, TrendingDownIcon, BarChart2Icon, TargetIcon, GridIcon, ClockIcon } from 'lucide-vue-next'
import axios from 'axios'
import { createChart } from 'lightweight-charts'

const status = ref({ running: false, daily_stats: { profit: 0 }, account: {} })
const activeTab = ref('COMMAND')
const positions = ref([])
const logs = ref([])
const history = ref([])
const marketData = ref([])
const historyData = ref([])
const advStats = ref({})
const currentTime = ref(new Date().toLocaleTimeString())

const selectedSymbol = ref(null)
const modelDetails = ref(null)
const showConfig = ref(false)
const configData = ref(null)
const chartContainer = ref(null)
const equityData = ref([])

let chart = null
let candleSeries = null
let equityChart = null
let equitySeries = null

const API_URL = 'http://localhost:8000/api'

const fetchStatus = async () => {
  try {
    const res = await axios.get(`${API_URL}/status`)
    status.value = res.data
  } catch (e) { console.error("API Error", e) }
}

const fetchMarket = async () => {
  try {
    const res = await axios.get(`${API_URL}/market`)
    marketData.value = res.data
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

const fetchStats = async () => {
  try {
    const res = await axios.get(`${API_URL}/stats`)
    advStats.value = res.data
    
    // Update Equity Chart
    if (res.data.equity_curve) {
        equityData.value = res.data.equity_curve.map((val, i) => ({ time: i, value: val }))
    }
  } catch (e) { console.error("API Error", e) }
}

const fetchHistory = async () => {
  try {
    const res = await axios.get(`${API_URL}/history`)
    historyData.value = res.data
  } catch (e) { console.error("API Error", e) }
}

const inspectModel = async (symbol) => {
  selectedSymbol.value = symbol
  modelDetails.value = null
  try {
    const res = await axios.get(`${API_URL}/model/${symbol}`)
    modelDetails.value = res.data
    
    // Init Chart
    await nextTick()
    if (chartContainer.value) {
        if (chart) chart.remove()
        chart = createChart(chartContainer.value, {
            layout: { background: { color: 'transparent' }, textColor: '#9ca3af' },
            grid: { vertLines: { color: '#374151' }, horzLines: { color: '#374151' } },
            width: chartContainer.value.clientWidth,
            height: chartContainer.value.clientHeight,
            timeScale: { timeVisible: true, secondsVisible: false },
        })
        candleSeries = chart.addCandlestickSeries({
            upColor: '#4ade80', downColor: '#f87171', borderVisible: false, wickUpColor: '#4ade80', wickDownColor: '#f87171'
        })
        
        // Mock Data (Replace with real OHLC later)
        const currentPrice = marketData.value.find(s => s.name === symbol)?.price || 0
        candleSeries.setData([
            { time: '2023-10-01', open: currentPrice-10, high: currentPrice+5, low: currentPrice-15, close: currentPrice-5 },
            { time: '2023-10-02', open: currentPrice-5, high: currentPrice+10, low: currentPrice-8, close: currentPrice+2 },
            { time: '2023-10-03', open: currentPrice+2, high: currentPrice+15, low: currentPrice, close: currentPrice+10 },
        ])
        chart.timeScale().fitContent()
    }

  } catch (e) { console.error("Model API Error", e) }
}

const openConfig = async () => {
  showConfig.value = true
  configData.value = null
  try {
    const res = await axios.get(`${API_URL}/config`)
    configData.value = res.data
  } catch (e) { console.error("Config API Error", e) }
}

const saveConfig = async () => {
  try {
    await axios.post(`${API_URL}/config`, configData.value)
    showConfig.value = false
    logs.value.unshift({ time: new Date().toLocaleTimeString(), type: 'SUCCESS', message: 'Configuration updated successfully' })
  } catch (e) {
    console.error("Config Save Error", e)
    alert("Failed to save config")
  }
}

const toggleBot = async () => {
  const action = status.value.running ? 'stop' : 'start'
  await axios.post(`${API_URL}/control/${action}`)
  await fetchStatus()
}

const evolve = async (symbol) => {
  await axios.post(`${API_URL}/control/evolve/${symbol}`)
}

onMounted(async () => {
  setInterval(fetchStatus, 1000)
  setInterval(fetchMarket, 1000)
  setInterval(fetchPositions, 1000)
  setInterval(fetchLogs, 2000)
  setInterval(fetchStats, 5000) // Fetch stats every 5s
  setInterval(fetchHistory, 5000) // Fetch history every 5s
  setInterval(() => {
    currentTime.value = new Date().toLocaleTimeString()
  }, 1000) // Update time every second
})

// Helpers
const formatMoney = (val) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val)

const getTrendColor = (trend) => {
  if (trend === 'BULL') return 'bg-green-500/20 text-green-400 border-green-500/30'
  if (trend === 'BEAR') return 'bg-red-500/20 text-red-400 border-red-500/30'
  return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
}

const getZScoreColor = (z) => {
  const val = parseFloat(z)
  if (Math.abs(val) > 2) return 'text-red-400 font-black'
  return 'text-gray-300'
}

const getLogColor = (type) => {
  if (type === 'ERROR') return 'text-red-400 font-bold'
  if (type === 'WARN') return 'text-yellow-400 font-bold'
  if (type === 'SUCCESS') return 'text-green-400 font-bold'
  return 'text-blue-300'
}
</script>

<style>
/* Custom Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #111827; }
::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4b5563; }
</style>
