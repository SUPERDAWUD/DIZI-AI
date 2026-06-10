/*
  main.js — Dashboard UI logic
  - Modular, non-emoji, enterprise-grade interactions
  - Streams logs/profile while orchestrator runs using SSE
  - Updates charts using Chart.js
*/

const api = {
  health: "/api/health",
  models: "/api/models",
  switchModel: "/api/switch-model",
  run: "/api/run-multi-agent",
  logs: "/api/logs",
  profile: "/api/profile",
  kvCache: "/api/kv-cache",
  recentRuns: "/api/recent-runs",
};

let runInProgress = false;
let eventSource = null;
let pollInterval = null;
let inputMode = 'prompt';
let lastPipelineResponse = null;

// Minimal wrapper for JSON fetch with consistent error handling
async function fetchJson(url, options = {}) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    const payload = await resp.json().catch(() => ({}));
    throw new Error(payload.error || `HTTP ${resp.status}`);
  }
  return resp.json();
}

/* UI helpers */
function el(id) { return document.getElementById(id); }

function setSystemStatus(text, level = 'neutral') {
  const elStatus = el('system-status');
  elStatus.querySelector('span').textContent = text;
  elStatus.dataset.level = level;
}

function appendStreamMessage(text, cls = 'info') {
  const container = el('stream');
  const node = document.createElement('div');
  node.className = `stream-item stream-${cls}`;
  const ts = new Date().toISOString();
  node.innerHTML = `<div class="stream-ts">${ts}</div><div class="stream-body">${text}</div>`;
  container.appendChild(node);
  container.scrollTop = container.scrollHeight;
}

function setRawJsonPanel(data) {
  const rawPanel = el('raw-json-panel');
  if (!rawPanel) return;
  rawPanel.textContent = JSON.stringify(data || {}, null, 2);
}

function clearOutput() {
  lastPipelineResponse = null;
  const bubble = el('output-bubble');
  if (bubble) {
    bubble.innerText = 'Waiting for output...';
    bubble.style.display = 'block';
  }

  const imageBubble = el('image-bubble');
  if (imageBubble) {
    imageBubble.innerHTML = '';
    imageBubble.style.display = 'none';
  }

  setRawJsonPanel({});
}

async function copyOutput() {
  const bubble = el('output-bubble');
  const text = bubble ? bubble.innerText.trim() : '';
  if (!text || text === 'Waiting for output...') return;

  try {
    await navigator.clipboard.writeText(text);
    appendStreamMessage('Output copied to clipboard.', 'success');
  } catch (err) {
    appendStreamMessage(`Copy failed: ${err.message}`, 'error');
  }
}

function toggleRawJson() {
  const toggle = el('show-raw-json');
  const rawPanel = el('raw-json-panel');
  if (!toggle || !rawPanel) return;
  rawPanel.hidden = !toggle.checked;
  if (toggle.checked) setRawJsonPanel(lastPipelineResponse || {});
}

function handleStreamEvent(event) {
  const type = event.event || 'event';
  switch (type) {
    case 'task_start':
      appendStreamMessage(`Task started: ${event.task_id} by ${event.agent}`, 'info');
      break;
    case 'task_complete':
      appendStreamMessage(`Task complete: ${event.task_id} (${event.task_type}) in ${event.duration.toFixed?.(2) || event.duration}s`, 'success');
      break;
    case 'kv_cache_update':
      appendStreamMessage(`KV cache updated (${Object.keys(event.kv_cache || {}).length} entries)`, 'log');
      el('kv-stats').textContent = `Entries: ${event.kv_cache?.entries || 0} | Reads: ${event.kv_cache?.reads || 0} | Writes: ${event.kv_cache?.writes || 0}`;
      break;
    case 'profile_update':
      appendStreamMessage('Profiler metrics updated', 'log');
      updateCharts(event.profile || {});
      break;
    case 'hardware_profile':
      appendStreamMessage('Hardware profile generated', 'log');
      updateCharts({ hw_metrics: event.profile || {} });
      break;
    case 'agent_not_found':
      appendStreamMessage(`Agent not found: ${event.agent_name}`, 'error');
      break;
    default:
      appendStreamMessage(JSON.stringify(event), 'log');
  }
}

function startStream() {
  if (eventSource) {
    return;
  }

  if (!window.EventSource) {
    appendStreamMessage('EventSource is unavailable in this browser; falling back to polling.', 'warn');
    startPolling();
    return;
  }

  eventSource = new EventSource('/api/stream');
  eventSource.onmessage = (e) => {
    try {
      const payload = JSON.parse(e.data);
      handleStreamEvent(payload);
    } catch (err) {
      console.debug('Invalid stream payload', err);
    }
  };
  eventSource.onerror = () => {
    appendStreamMessage('Realtime event stream disconnected.', 'warn');
    stopStream();
  };
}

function stopStream() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  stopPolling();
}

/* Agent status list */
function setInputMode(mode) {
  inputMode = mode;
  const promptButton = el('mode-prompt');
  const fileButton = el('mode-file');
  const textInput = el('user-input');
  const filePicker = el('file-picker-label');
  const fileInput = el('file-input');

  if (mode === 'prompt') {
    promptButton.classList.add('active');
    fileButton.classList.remove('active');
    textInput.classList.remove('disabled-input');
    textInput.disabled = false;
    filePicker.classList.add('disabled');
    fileInput.disabled = true;
    fileInput.value = '';
  } else {
    promptButton.classList.remove('active');
    fileButton.classList.add('active');
    textInput.classList.add('disabled-input');
    textInput.disabled = true;
    filePicker.classList.remove('disabled');
    fileInput.disabled = false;
  }
}

function setAgentList(agents) {
  const list = el('agent-list');
  list.innerHTML = '';
  agents.forEach(a => {
    const li = document.createElement('li');
    li.className = 'agent-item';
    li.dataset.agent = a.name;
    li.innerHTML = `<div class="agent-name">${a.name}</div><div class="agent-status status-${a.status}"></div>`;
    list.appendChild(li);
  });
}

/* Charts */
let chartLatency = null;
let chartThroughput = null;
let chartKV = null;
let chartMemory = null;
let chartHW = null;
function initCharts() {
  const ctxL = el('chart-latency').getContext('2d');
  chartLatency = new Chart(ctxL, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Latency (ms)', data: [], borderColor: '#5EEAD4', tension: 0.25, pointRadius:0 }] },
    options: { responsive: true, plugins:{legend:{display:false}}, scales:{x:{display:false}} }
  });

  const ctxT = el('chart-throughput').getContext('2d');
  chartThroughput = new Chart(ctxT, {
    type: 'bar',
    data: { labels: [], datasets: [{ label: 'Tokens/sec', data: [], backgroundColor: '#60A5FA' }] },
    options: { responsive:true, plugins:{legend:{display:false}}, scales:{x:{display:false}} }
  });

  // KV growth chart (line)
  const ctxK = el('chart-kv').getContext('2d');
  chartKV = new Chart(ctxK, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'KV entries', data: [], borderColor: '#f59e0b', tension: 0.25, pointRadius:0 }] },
    options: { responsive:true, plugins:{legend:{display:false}}, scales:{x:{display:false}} }
  });

  // Memory usage chart (area)
  const ctxM = el('chart-memory').getContext('2d');
  chartMemory = new Chart(ctxM, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Memory (MB)', data: [], backgroundColor: 'rgba(96,165,250,0.12)', borderColor:'#60A5FA', fill:true, tension:0.3, pointRadius:0 }] },
    options: { responsive:true, plugins:{legend:{display:false}}, scales:{x:{display:false}} }
  });
}

function updateChartsLegacy(profile) {
  return updateCharts(profile);
  if (!profile) return;
  // profile expected to have per-agent latency and tokens/sec arrays
  const latencySamples = profile.latency_samples || [];
  const tps = profile.tokens_per_sec || [];
  const kvGrowth = profile.kv_growth || [];
  const memorySamples = profile.memory_samples || [];
  const hwMetrics = profile.hw_metrics || {};

  chartLatency.data.labels = latencySamples.map((_,i) => i+1);
  chartLatency.data.datasets[0].data = latencySamples;
  chartLatency.update();

  chartThroughput.data.labels = tps.map((_,i) => i+1);
  chartThroughput.data.datasets[0].data = tps;
  chartThroughput.update();
  const tokenRate = tps.length ? `${tps[tps.length - 1]} tokens/s` : '—';
  const tokenRateElement = document.getElementById('token-rate');
  if (tokenRateElement) tokenRateElement.textContent = tokenRate;

  if (chartKV) {
    chartKV.data.labels = kvGrowth.map((_,i) => i+1);
    chartKV.data.datasets[0].data = kvGrowth;
    chartKV.update();
  }

  if (chartMemory) {
    chartMemory.data.labels = memorySamples.map((_,i) => i+1);
    chartMemory.data.datasets[0].data = memorySamples;
    chartMemory.update();
  }

  // Update hardware text metrics if present
  if (hwMetrics) {
    if (hwMetrics.matmul_tiles !== undefined) el('hw-tiles').textContent = `MatMul tiles: ${hwMetrics.matmul_tiles}`;
    if (hwMetrics.sram !== undefined) el('hw-sram').textContent = `SRAM: ${hwMetrics.sram}`;
    if (hwMetrics.bandwidth !== undefined) el('hw-bandwidth').textContent = `Bandwidth: ${hwMetrics.bandwidth}`;
    if (hwMetrics.predicted_tps !== undefined) el('hw-predicted').textContent = `Predicted tokens/sec: ${hwMetrics.predicted_tps}`;
  }
}

function updateCharts(profile) {
  if (!profile) return;

  const hwMetrics = profile.hw_metrics || {};

  if (profile.total_time_seconds !== undefined || profile.total_tokens !== undefined || profile.task_count !== undefined || Array.isArray(profile.tasks)) {
    const tasks = Array.isArray(profile.tasks) ? profile.tasks : [];
    const totalTime = Number(profile.total_time_seconds || 0);
    const totalTokens = Number(profile.total_tokens || 0);
    const taskCount = Number(profile.task_count || tasks.length || 0);
    const latestMemory = tasks.length ? Number(tasks[tasks.length - 1].memory_rss_mb || 0) : 0;
    const memoryValue = latestMemory || tasks.length;

    if (chartLatency) {
      chartLatency.data.labels = ['Total'];
      chartLatency.data.datasets[0].data = [totalTime];
      chartLatency.update();
    }

    if (chartThroughput) {
      chartThroughput.data.labels = ['Total'];
      chartThroughput.data.datasets[0].data = [totalTokens];
      chartThroughput.update();
    }

    const tokenRateElement = document.getElementById('token-rate');
    if (tokenRateElement) tokenRateElement.textContent = `${totalTokens} tokens`;

    if (chartKV) {
      chartKV.data.labels = ['Tasks'];
      chartKV.data.datasets[0].data = [taskCount];
      chartKV.update();
    }

    if (chartMemory) {
      chartMemory.data.labels = ['Latest'];
      chartMemory.data.datasets[0].data = [memoryValue];
      chartMemory.update();
    }
  }

  if (hwMetrics) {
    if (hwMetrics.matmul_tiles !== undefined) el('hw-tiles').textContent = `MatMul tiles: ${hwMetrics.matmul_tiles}`;
    if (hwMetrics.sram !== undefined) el('hw-sram').textContent = `SRAM: ${hwMetrics.sram}`;
    if (hwMetrics.bandwidth !== undefined) el('hw-bandwidth').textContent = `Bandwidth: ${hwMetrics.bandwidth}`;
    if (hwMetrics.predicted_tps !== undefined) el('hw-predicted').textContent = `Predicted tokens/sec: ${hwMetrics.predicted_tps}`;
  }
}

/* Poll endpoints while run is in progress */
async function startPolling() {
  if (pollInterval) return;
  pollInterval = setInterval(async () => {
    try {
      const logs = await fetchJson(api.logs);
      const profile = await fetchJson(api.profile);
      const kv = await fetchJson(api.kvCache);

      // update stream with latest logs (replace whole for simplicity)
      if (Array.isArray(logs.logs)) {
        // show only last 20
        const last = logs.logs.slice(-20);
        el('stream').innerHTML = '';
        last.forEach(e => appendStreamMessage(JSON.stringify(e), 'log'));
      }

      // update charts and kv
      updateCharts(profile.profile || profile);
      const kvs = (kv.kv_cache) ? kv.kv_cache : kv;
      el('kv-stats').textContent = `Entries: ${kvs.entries||0} | Reads: ${kvs.reads||0} | Writes: ${kvs.writes||0}`;

    } catch (err) {
      // polling errors are non-fatal for UI
      console.debug('poll error', err);
    }
  }, 800);
}

function stopPolling() {
  if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
}

/* Run orchestrator with file upload or filepath */
async function runOrchestrator({ file=null, prompt='', useFake=true } = {}) {
  runInProgress = true;
  setSystemStatus('running','running');
  appendStreamMessage('Orchestrator starting', 'info');
  startStream();
  // Clear previous outputs
  const outEl = el('output-bubble'); if (outEl) outEl.textContent = '';
  const imgEl = el('image-bubble'); if (imgEl) imgEl.innerHTML = '';

  try {
    const form = new FormData();
    form.append('mode', inputMode);

    if (inputMode === 'prompt') {
      const trimmed = String(prompt || '').trim();
      if (!trimmed) {
        appendStreamMessage('Prompt mode requires text input.', 'error');
        setSystemStatus('idle','neutral');
        runInProgress = false;
        stopStream();
        return;
      }
      form.append('prompt', trimmed);
    } else {
      if (!file) {
        appendStreamMessage('File mode requires an uploaded .txt file.', 'error');
        setSystemStatus('idle','neutral');
        runInProgress = false;
        stopStream();
        return;
      }
      form.append('file', file);
      form.append('prompt', '');
    }

    form.append('use_fake_hardware', useFake ? 'true' : 'false');

    const resp = await fetch(api.run, { method: 'POST', body: form });
    const result = await resp.json();
    console.log("DATA RECEIVED:", result);

    if (resp.ok) {
      appendStreamMessage('Orchestrator finished', 'success');
      updateCharts(result.profile || {});
      el('kv-stats').textContent = `Entries: ${result.kv_cache?.entries||0} | Reads: ${result.kv_cache?.reads||0} | Writes: ${result.kv_cache?.writes||0}`;
      
      // Put textual output into the output bubble
      if (result.output) {
        const outBubble = document.getElementById('output-bubble');
        if (outBubble) {
          outBubble.innerText = result.output;
          outBubble.style.display = 'block';
        }
      }
      
      // Insert image output into the image bubble
      if (result.image_url) {
        const imgBubble = document.getElementById('image-bubble');
        if (imgBubble) {
          imgBubble.innerHTML = `<img src="${result.image_url}" />`;
          imgBubble.style.display = 'block';
        }
      }
      
      // Also support result.images array and result.image for backwards compatibility
      if ((result.images && Array.isArray(result.images) && result.images.length > 0) || result.image) {
        const imageBubble = document.getElementById('image-bubble');
        if (imageBubble) {
          imageBubble.innerHTML = '';
          const imgs = [];
          if (result.images && Array.isArray(result.images)) imgs.push(...result.images);
          if (result.image) imgs.push(result.image);
          imgs.forEach(src => {
            try {
              const i = document.createElement('img');
              i.src = src;
              i.alt = 'generated image';
              imageBubble.appendChild(i);
            } catch (e) {
              console.debug('invalid image source', e);
            }
          });
          imageBubble.style.display = 'block';
        }
      }
    } else {
      appendStreamMessage(`Run failed: ${result.error || 'unknown'}`, 'error');
    }

  } catch (err) {
    console.error("FETCH ERROR:", err);
    appendStreamMessage(`Execution error: ${err.message}`, 'error');
  } finally {
    runInProgress = false;
    setSystemStatus('idle','neutral');
    stopStream();
  }
}

/* Wire UI events */
function attachEvents() {
  el('top-run').addEventListener('click', () => {
    runPipeline();
  });

  el('mode-prompt').addEventListener('click', () => setInputMode('prompt'));
  el('mode-file').addEventListener('click', () => setInputMode('file'));
  el('agent-mode').addEventListener('change', () => {
    if (el('agent-mode').value !== 'pipeline') {
      setInputMode('prompt');
    }
  });
  el('clear-output').addEventListener('click', clearOutput);
  el('copy-output').addEventListener('click', copyOutput);
  el('show-raw-json').addEventListener('change', toggleRawJson);

  el('top-stop').addEventListener('click', () => {
    // Stop polling and update UI (server-side cancellation not implemented)
    stopPolling();
    runInProgress = false;
    setSystemStatus('stopped','neutral');
    appendStreamMessage('Run stopped (UI only)', 'info');
  });
}

/* Initial load */
async function init() {
  initCharts();
  attachEvents();
  setSystemStatus('ready','neutral');

  // Load models list into both top selector and left manager
  try {
    const data = await fetchJson(api.models);
    const models = data.models || {};
    const top = el('top-model-select');
    top.innerHTML = '';
    Object.keys(models).forEach(k => { const o = document.createElement('option'); o.value = k; o.textContent = k; top.appendChild(o); });

    // initialize agents (placeholder statuses)
    setAgentList([
      { name: 'ReaderAgent', status: 'idle' },
      { name: 'SummarizerAgent', status: 'idle' },
      { name: 'CheckerAgent', status: 'idle' },
    ]);
  } catch (err) {
    appendStreamMessage('Failed to initialize models: '+err.message, 'error');
  }

  setInputMode('prompt');
}

document.addEventListener('DOMContentLoaded', init);

async function runPipeline() {
    console.log("RUN PIPELINE FIRED");

    const input = document.getElementById("user-input").value;
    const mode = document.getElementById("agent-mode").value;
    setSystemStatus('running', 'running');

    try {
        const response = await fetch("/api/run-multi-agent", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt: input, mode, extra: null })
        });

        const data = await response.json();
        lastPipelineResponse = data;
        setRawJsonPanel(data);
        console.log("FULL RESPONSE:", data);

        const bubble = document.getElementById("output-bubble");
        bubble.innerText = data.output || "No output returned.";
        bubble.style.display = "block";

        const imageBubble = document.getElementById("image-bubble");
        if (imageBubble) {
            if (data.image_url) {
                imageBubble.innerHTML = `<img src="${data.image_url}" alt="Generated image" />`;
                imageBubble.style.display = "block";
            } else {
                imageBubble.innerHTML = "";
                imageBubble.style.display = "none";
            }
        }

        // Update charts
        if (data.profile) {
            updateCharts(data.profile);
        }

        if (!response.ok) {
            appendStreamMessage(`Run failed: ${data.output || response.status}`, 'error');
        } else {
            appendStreamMessage(`Run complete in ${mode} mode.`, 'success');
        }
    } catch (err) {
        lastPipelineResponse = { output: `Request failed: ${err.message}` };
        setRawJsonPanel(lastPipelineResponse);
        const bubble = document.getElementById("output-bubble");
        bubble.innerText = lastPipelineResponse.output;
        bubble.style.display = "block";
        appendStreamMessage(lastPipelineResponse.output, 'error');
    } finally {
        setSystemStatus('ready', 'neutral');
    }
}

window.runPipeline = runPipeline;
