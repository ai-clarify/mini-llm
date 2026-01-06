const sections = document.querySelectorAll('.page-section');
const navButtons = document.querySelectorAll('.nav-btn');
const state = {
  runs: [],
  jobs: [],
  selectedRun: null,
  selectedJob: null,
};
let runChart = null;
let autoRefreshTimer = null;

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || `请求失败: ${res.status}`);
  }
  return res.json();
}

function switchSection(name) {
  sections.forEach((section) => {
    if (section.dataset.section === name) {
      section.classList.add('active');
    } else {
      section.classList.remove('active');
    }
  });
  navButtons.forEach((btn) => {
    if (btn.dataset.section === name) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
}

function fmtBytes(bytes) {
  if (bytes === undefined || bytes === null) return '未知';
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let idx = 0;
  while (size >= 1024 && idx < units.length - 1) {
    size /= 1024;
    idx += 1;
  }
  return `${size.toFixed(size < 10 ? 1 : 0)} ${units[idx]}`;
}

function fmtLines(count, capped) {
  if (!count) return '未知';
  return capped ? `${count}+` : `${count}`;
}

function fmtNumber(value) {
  if (value === undefined || value === null || Number.isNaN(value)) return '未知';
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(2)}k`;
  return String(value);
}

function fmtMetric(value) {
  if (value === undefined || value === null || Number.isNaN(value)) return '未知';
  if (Number.isInteger(value)) return value.toString();
  return value.toFixed(4);
}

function renderOverview(data) {
  const grid = document.getElementById('overview-grid');
  grid.innerHTML = '';
  const cards = [
    { label: '数据集', value: data.datasets },
    { label: '运行记录', value: data.runs },
    { label: '配置模板', value: data.configs },
  ];
  cards.forEach((item) => {
    const div = document.createElement('div');
    div.className = 'stat-card';
    div.innerHTML = `
      <div class="stat-label">${item.label}</div>
      <div class="stat-value">${item.value ?? 0}</div>
      <div class="meta">实时刷新</div>
    `;
    grid.appendChild(div);
  });

  const tag = document.getElementById('latest-run-tag');
  if (data.latest_run) {
    const run = data.latest_run;
    tag.textContent = `最近运行 · ${run.name} (${run.stage})`;
  } else {
    tag.textContent = '最近运行：暂无记录';
  }
}

function renderDatasets(rows) {
  const tbody = document.getElementById('dataset-table-body');
  tbody.innerHTML = '';
  if (!rows.length) {
    const empty = document.createElement('tr');
    empty.innerHTML = '<td colspan="5" class="meta">未找到数据文件</td>';
    tbody.appendChild(empty);
    return;
  }
  rows.forEach((row) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><input type="checkbox" data-path="${row.path}"></td>
      <td><code>${row.path}</code></td>
      <td>${fmtLines(row.line_count, row.line_count_capped)}</td>
      <td>${fmtBytes(row.size_bytes)}</td>
      <td class="muted">${row.preview.slice(0, 2).map((p) => p.replace(/</g, '&lt;')).join('<br>')}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderConfigs(configs) {
  const grid = document.getElementById('config-grid');
  grid.innerHTML = '';
  if (!configs.length) {
    grid.innerHTML = '<div class="meta">暂无配置模板</div>';
    return;
  }
  configs.forEach((cfg) => {
    const card = document.createElement('div');
    card.className = 'config-card';
    const meta = cfg.content.meta || {};
    const stage = meta.stage || 'pipeline';
    const runnable = ['pretrain', 'sft', 'dpo', 'pipeline'].includes(stage);
    card.innerHTML = `
      <div class="config-head">
        <div>
          <div class="config-name">${cfg.name}</div>
          <div class="meta">${cfg.description || '可直接用于训练脚本或 run.sh/MLX 管线'}</div>
        </div>
        <div class="config-actions">
          ${runnable ? `<button class="primary start-train" data-config-path="${cfg.path}">开始训练</button>` : ''}
          <span class="tag">${stage}</span>
        </div>
      </div>
      <div class="meta">版本：${meta.version || '未标注'}</div>
      <details class="config-body">
        <summary>查看内容 (${cfg.path})</summary>
        <pre>${JSON.stringify(cfg.content, null, 2)}</pre>
      </details>
    `;
    const startBtn = card.querySelector('.start-train');
    if (startBtn) {
      startBtn.addEventListener('click', async (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        await startTraining(cfg.path, startBtn);
      });
    }
    grid.appendChild(card);
  });
}

function runBadge(run) {
  const parts = [];
  if (run.stage && run.stage !== 'unknown') parts.push(run.stage);
  parts.push(run.kind === 'mlx' ? 'MLX' : 'Torch');
  return parts.join(' · ');
}

function renderRuns(runs) {
  state.runs = runs || [];
  const list = document.getElementById('runs-list');
  list.innerHTML = '';
  if (!state.runs.length) {
    list.innerHTML = '<div class="meta">暂无运行记录</div>';
    return;
  }
  state.runs.forEach((run) => {
    const div = document.createElement('div');
    div.className = 'list-item';
    div.innerHTML = `
      <div class="item-head">
        <div>
          <div class="item-title">${run.name}</div>
          <div class="meta">${runBadge(run)}</div>
        </div>
        <span class="tag">${new Date(run.modified_at).toLocaleString()}</span>
      </div>
      <div class="meta">${run.latest_checkpoint ? `Checkpoint: ${run.latest_checkpoint}` : '暂无权重'}</div>
      <div class="metric-row">${Object.entries(run.metrics || {}).map(([k, v]) => `<span class="pill">${k}: ${fmtMetric(v)}</span>`).join(' ') || '<span class="meta">暂无标量</span>'}</div>
    `;
    div.addEventListener('click', () => {
      document.querySelectorAll('#runs-list .list-item').forEach((el) => el.classList.remove('active'));
      div.classList.add('active');
      state.selectedJob = null;
      state.selectedRun = run.id;
      showRunDetail(run.id);
    });
    if (state.selectedRun && run.id === state.selectedRun) {
      div.classList.add('active');
    }
    list.appendChild(div);
  });
  const first = list.querySelector('.list-item');
  if (first && !state.selectedJob && !document.querySelector('#runs-list .list-item.active')) {
    first.classList.add('active');
    state.selectedRun = state.runs[0].id;
    showRunDetail(state.runs[0].id);
  }
}

function jobBadge(job) {
  const parts = [];
  if (job.stage) parts.push(job.stage);
  if (job.kind) parts.push(job.kind);
  return parts.join(' · ');
}

function renderJobs(jobs) {
  state.jobs = jobs || [];
  const list = document.getElementById('jobs-list');
  list.innerHTML = '';
  if (!state.jobs.length) {
    list.innerHTML = '<div class="meta">暂无任务</div>';
    return;
  }
  state.jobs.forEach((job) => {
    const div = document.createElement('div');
    div.className = 'list-item';
    const stateLabel = job.state || 'unknown';
    const started = job.started_at ? new Date(job.started_at).toLocaleString() : '未开始';
    div.innerHTML = `
      <div class="item-head">
        <div>
          <div class="item-title">${job.config_path.split('/').slice(-1)[0]} <span class="pill">${stateLabel}</span></div>
          <div class="meta">${jobBadge(job)} · ${started}</div>
        </div>
        <span class="tag">${job.pid ? `PID ${job.pid}` : ''}</span>
      </div>
      <div class="meta">${job.run_id ? `Run: <code>${job.run_id}</code>` : ''}</div>
    `;

    div.addEventListener('click', () => {
      document.querySelectorAll('#jobs-list .list-item').forEach((el) => el.classList.remove('active'));
      div.classList.add('active');
      state.selectedJob = job.id;
      state.selectedRun = job.run_id || null;
      showJobDetail(job.id);
      if (job.run_id) showRunDetail(job.run_id);
    });

    if (state.selectedJob && job.id === state.selectedJob) {
      div.classList.add('active');
    }
    list.appendChild(div);
  });
}

function applyRunSummary(detail) {
  const meta = document.getElementById('run-meta');
  const summary = document.getElementById('run-summary');
  const run = detail.run;
  meta.textContent = `${run.name} · ${runBadge(run)} · ${new Date(run.modified_at).toLocaleString()}`;
  const metrics = run.metrics && Object.keys(run.metrics).length
    ? Object.entries(run.metrics).map(([k, v]) => `<span class="pill">${k}: ${fmtMetric(v)}</span>`).join(' ')
    : '<span class="meta">暂无标量</span>';
  summary.innerHTML = `
    <div class="meta">Checkpoint: ${run.latest_checkpoint || '未找到'}</div>
    <div class="metric-row">${metrics}</div>
  `;
}

function applyMlxSummary(detail) {
  const target = document.getElementById('mlx-summary');
  const mlx = detail.mlx;
  if (!mlx) {
    target.textContent = '';
    return;
  }
  const parts = [];
  if (mlx.task) parts.push(`任务: ${mlx.task}`);
  if (mlx.step !== undefined) parts.push(`步数: ${fmtNumber(mlx.step)}`);
  if (mlx.seen_tokens !== undefined) parts.push(`Seen tokens: ${fmtNumber(mlx.seen_tokens)}`);
  if (mlx.preset) parts.push(`预设: ${mlx.preset}`);
  const model = mlx.model || {};
  const modelBits = [];
  if (model.hidden_size) modelBits.push(`D${model.hidden_size}`);
  if (model.num_hidden_layers) modelBits.push(`${model.num_hidden_layers}L`);
  if (model.num_attention_heads) modelBits.push(`${model.num_attention_heads}H`);
  if (model.num_key_value_heads) modelBits.push(`KV${model.num_key_value_heads}`);
  target.innerHTML = `
    <div class="pill">MLX 运行</div>
    <div class="meta">${parts.join(' · ')}</div>
    <div class="meta">模型: ${modelBits.join(' / ') || '未知'}</div>
    <div class="meta">Checkpoint 目录: ${mlx.checkpoint_dir || '未找到'}</div>
  `;
}

function drawRunChart(series) {
  const canvas = document.getElementById('run-chart');
  const status = document.getElementById('chart-status');
  if (runChart) {
    runChart.destroy();
    runChart = null;
  }
  const entries = Object.entries(series || {}).filter(([, values]) => values && values.length);
  if (!entries.length) {
    if (status) status.textContent = '暂无标量 (TensorBoard 日志未找到)';
    return;
  }
  if (status) status.textContent = '';
  const datasets = entries.map(([tag, values], idx) => ({
    label: tag,
    data: values.map((v) => ({ x: v.step, y: v.value })),
    borderWidth: 2,
    fill: false,
    borderColor: ['#cbd5e1', '#94a3b8', '#e5e7eb', '#9ca3af'][idx % 4],
    backgroundColor: 'rgba(229, 231, 235, 0.12)',
  }));
  runChart = new Chart(canvas, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#cbd5e1' } } },
      scales: {
        x: { type: 'linear', ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.1)' } },
        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.1)' } },
      },
    },
  });
}

async function showRunDetail(runId) {
  try {
    const detail = await fetchJSON(`/api/runs/${encodeURIComponent(runId)}`);
    applyRunSummary(detail);
    applyMlxSummary(detail);
    drawRunChart(detail.scalars || {});
  } catch (err) {
    const meta = document.getElementById('run-meta');
    if (meta) meta.textContent = `加载失败: ${err.message}`;
  }
}

async function showJobDetail(jobId) {
  const stopBtn = document.getElementById('stop-job');
  const metaEl = document.getElementById('job-meta');
  const logEl = document.getElementById('job-log');
  if (!jobId) {
    if (metaEl) metaEl.textContent = '未选择任务';
    if (logEl) logEl.textContent = '';
    if (stopBtn) stopBtn.disabled = true;
    return;
  }
  try {
    const res = await fetchJSON(`/api/jobs/${encodeURIComponent(jobId)}?max_bytes=96000`);
    const job = res.job;
    const stateLabel = job.state || 'unknown';
    const bits = [];
    bits.push(`任务: ${job.config_path}`);
    bits.push(`状态: ${stateLabel}`);
    if (job.pid) bits.push(`PID: ${job.pid}`);
    if (job.started_at) bits.push(`开始: ${new Date(job.started_at).toLocaleString()}`);
    if (job.finished_at) bits.push(`结束: ${new Date(job.finished_at).toLocaleString()}`);
    if (job.run_id) bits.push(`Run: ${job.run_id}`);
    if (metaEl) metaEl.textContent = bits.join(' · ');
    if (logEl) logEl.textContent = res.log_tail || '';

    if (stopBtn) {
      stopBtn.disabled = !(stateLabel === 'running' || stateLabel === 'queued');
      stopBtn.onclick = async () => {
        stopBtn.disabled = true;
        try {
          await fetchJSON(`/api/jobs/${encodeURIComponent(jobId)}/stop`, { method: 'POST' });
          await refreshRunsAndJobs();
          await showJobDetail(jobId);
        } catch (err) {
          alert(`停止失败：${err.message}`);
        }
      };
    }
  } catch (err) {
    if (metaEl) metaEl.textContent = `任务加载失败: ${err.message}`;
    if (logEl) logEl.textContent = '';
    if (stopBtn) stopBtn.disabled = true;
  }
}

async function startTraining(configPath, btnEl) {
  if (!configPath) return;
  const btn = btnEl || null;
  if (btn) {
    btn.disabled = true;
    btn.textContent = '启动中...';
  }
  try {
    const job = await fetchJSON('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config_path: configPath }),
    });
    switchSection('runs');
    state.selectedJob = job.id;
    state.selectedRun = job.run_id || null;
    await refreshRunsAndJobs();
    await showJobDetail(job.id);
    if (job.run_id) await showRunDetail(job.run_id);
  } catch (err) {
    alert(`启动失败：${err.message}`);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = '开始训练';
    }
  }
}

async function buildSnapshot() {
  const btn = document.getElementById('build-snapshot');
  const selected = Array.from(document.querySelectorAll('input[type="checkbox"][data-path]:checked')).map((el) => el.dataset.path);
  const name = document.getElementById('snapshot-name').value.trim();
  if (!selected.length) {
    alert('请至少选择一个数据文件');
    return;
  }
  btn.disabled = true;
  btn.textContent = '生成中...';
  try {
    const res = await fetchJSON('/api/datasets/materialize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, files: selected }),
    });
    alert(`快照已生成：${res.combined_path}\n行数：${res.line_count}`);
    await refreshAll();
  } catch (err) {
    alert(`生成失败：${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = '生成快照';
  }
}

async function refreshAll() {
  const [overview, datasets, configs, runs] = await Promise.all([
    fetchJSON('/api/overview'),
    fetchJSON('/api/datasets'),
    fetchJSON('/api/configs'),
    fetchJSON('/api/runs'),
  ]);
  renderOverview(overview);
  renderDatasets(datasets);
  renderConfigs(configs);
  renderRuns(runs);
}

async function refreshRunsAndJobs() {
  const [jobs, runs] = await Promise.all([fetchJSON('/api/jobs'), fetchJSON('/api/runs')]);
  renderJobs(jobs);
  renderRuns(runs);
}

function setupNav() {
  navButtons.forEach((btn) => {
    btn.addEventListener('click', () => switchSection(btn.dataset.section));
  });
}

function setupSnapshotBuilder() {
  const btn = document.getElementById('build-snapshot');
  btn.addEventListener('click', buildSnapshot);
}

function setupRunsPanel() {
  const refreshBtn = document.getElementById('refresh-runs');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', async () => {
      refreshBtn.disabled = true;
      refreshBtn.textContent = '刷新中...';
      try {
        await refreshRunsAndJobs();
        if (state.selectedJob) await showJobDetail(state.selectedJob);
        if (state.selectedRun) await showRunDetail(state.selectedRun);
      } finally {
        refreshBtn.disabled = false;
        refreshBtn.textContent = '刷新';
      }
    });
  }
}

function startAutoRefresh() {
  if (autoRefreshTimer) return;
  autoRefreshTimer = setInterval(async () => {
    const active = document.querySelector('.page-section.active');
    if (!active || active.dataset.section !== 'runs') return;
    try {
      await refreshRunsAndJobs();
      if (state.selectedJob) await showJobDetail(state.selectedJob);
      if (state.selectedRun) await showRunDetail(state.selectedRun);
    } catch (err) {
      // ignore transient refresh errors
    }
  }, 3000);
}

document.addEventListener('DOMContentLoaded', async () => {
  setupNav();
  setupSnapshotBuilder();
  setupRunsPanel();
  await refreshAll();
  await refreshRunsAndJobs();
  startAutoRefresh();
});
