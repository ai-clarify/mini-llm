from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar

import mlx.core as mx


@dataclass(frozen=True)
class TraceConfig:
    """
    Activation tracing configuration for MLX inference.

    Notes:
    - "Parameters activated" in dense Transformers is effectively "all parameters";
      this tracer instead captures per-token activation magnitudes (RMS) at key points.
    """

    record_qkv: bool = False
    record_qkv_per_head: bool = False
    mlp_topk: int = 0
    record_attn: bool = False
    record_attn_all_queries: bool = False
    attn_topk: int = 16


T = TypeVar("T")


def _ensure_len(xs: List[Optional[T]], n: int, *, fill: Optional[T] = None) -> None:
    if len(xs) < n:
        xs.extend([fill] * (n - len(xs)))


def _token_rms(x: mx.array) -> mx.array:
    # x: [B, T, ...] -> rms over last dim
    x32 = x.astype(mx.float32)
    return mx.sqrt(mx.mean(x32 * x32, axis=-1))


def _token_rms_qkv(x: mx.array) -> mx.array:
    # x: [B, heads, T, head_dim] -> rms over head_dim => [B, heads, T]
    x32 = x.astype(mx.float32)
    return mx.sqrt(mx.mean(x32 * x32, axis=-1))


def _topk_with_indices(x: mx.array, *, k: int) -> Tuple[mx.array, mx.array]:
    """
    Return (values, indices) for the top-k largest elements along the last axis.

    MLX's `mx.topk` currently returns only the values, so we build indices via argpartition.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    k = int(k)
    last_dim = int(x.shape[-1])
    if k > last_dim:
        k = last_dim

    idx = mx.argpartition(-x, kth=k - 1, axis=-1)[..., :k]
    vals = mx.take_along_axis(x, idx, axis=-1)
    order = mx.argsort(-vals, axis=-1)
    idx = mx.take_along_axis(idx, order, axis=-1)
    vals = mx.take_along_axis(vals, order, axis=-1)
    return vals, idx


@dataclass
class LayerTrace:
    metrics: Dict[str, List[Optional[float]]] = field(default_factory=dict)
    mlp_topk: List[Optional[Dict[str, Any]]] = field(default_factory=list)
    qkv_rms_by_head: Dict[str, List[List[Optional[float]]]] = field(default_factory=dict)
    # For each token position: list[head] -> {"idx": [...], "w": [...]}
    attn_topk: List[Optional[List[Dict[str, Any]]]] = field(default_factory=list)


class ActivationTracer:
    def __init__(self, *, num_layers: int, cfg: TraceConfig):
        self.cfg = cfg
        self.created_at_s = time.time()
        self.token_ids: List[Optional[int]] = []
        self.layers: List[LayerTrace] = [LayerTrace() for _ in range(int(num_layers))]
        self.num_heads: Optional[int] = None

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    def on_input_ids(self, *, start_pos: int, input_ids: mx.array) -> None:
        # input_ids: [1, T] during inference.
        ids = input_ids.astype(mx.int32)
        mx.eval(ids)
        ids_list: List[int] = ids[0].tolist()
        n = int(start_pos) + len(ids_list)
        _ensure_len(self.token_ids, n, fill=None)
        for i, tid in enumerate(ids_list):
            self.token_ids[int(start_pos) + i] = int(tid)

        for layer in self.layers:
            for name, values in layer.metrics.items():
                _ensure_len(values, n, fill=None)
            if int(self.cfg.mlp_topk) > 0:
                _ensure_len(layer.mlp_topk, n, fill=None)
            if bool(self.cfg.record_attn):
                _ensure_len(layer.attn_topk, n, fill=None)

    def _record_token_metric(self, *, layer_id: int, name: str, start_pos: int, values_bt: mx.array) -> None:
        # values_bt: [B, T] (B=1)
        mx.eval(values_bt)
        values_t: List[float] = values_bt[0].tolist()
        layer = self.layers[int(layer_id)]
        series = layer.metrics.get(name)
        if series is None:
            series = []
            layer.metrics[name] = series
        n = int(start_pos) + len(values_t)
        _ensure_len(series, n, fill=None)
        for i, v in enumerate(values_t):
            series[int(start_pos) + i] = float(v)

    def record_hidden(self, *, layer_id: int, name: str, start_pos: int, x: mx.array) -> None:
        self._record_token_metric(layer_id=int(layer_id), name=str(name), start_pos=int(start_pos), values_bt=_token_rms(x))

    def record_qkv(self, *, layer_id: int, start_pos: int, q: mx.array, k: mx.array, v: mx.array) -> None:
        if not self.cfg.record_qkv:
            return

        q_rms = _token_rms_qkv(q)  # [B, H, T]
        k_rms = _token_rms_qkv(k)  # [B, Hkv, T]
        v_rms = _token_rms_qkv(v)

        # Aggregate to a single scalar per token for heatmaps.
        self._record_token_metric(
            layer_id=int(layer_id), name="q_rms", start_pos=int(start_pos), values_bt=mx.mean(q_rms, axis=1)
        )
        self._record_token_metric(
            layer_id=int(layer_id), name="k_rms", start_pos=int(start_pos), values_bt=mx.mean(k_rms, axis=1)
        )
        self._record_token_metric(
            layer_id=int(layer_id), name="v_rms", start_pos=int(start_pos), values_bt=mx.mean(v_rms, axis=1)
        )

        if not self.cfg.record_qkv_per_head:
            return

        def _record_by_head(key: str, rms_bht: mx.array) -> None:
            mx.eval(rms_bht)
            rms_ht: List[List[float]] = rms_bht[0].tolist()  # [heads][T]
            layer = self.layers[int(layer_id)]
            by_head = layer.qkv_rms_by_head.get(key)
            if by_head is None:
                by_head = []
                layer.qkv_rms_by_head[key] = by_head
            _ensure_len(by_head, len(rms_ht), fill=[])
            for h, series in enumerate(rms_ht):
                n = int(start_pos) + len(series)
                if len(by_head[h]) < n:
                    by_head[h].extend([None] * (n - len(by_head[h])))
                for i, val in enumerate(series):
                    by_head[h][int(start_pos) + i] = float(val)

        _record_by_head("q_rms_by_head", q_rms)
        _record_by_head("k_rms_by_head", k_rms)
        _record_by_head("v_rms_by_head", v_rms)

    def record_mlp_act(self, *, layer_id: int, start_pos: int, act: mx.array) -> None:
        # act: [B, T, I]
        self.record_hidden(layer_id=int(layer_id), name="mlp_act_rms", start_pos=int(start_pos), x=act)

        k = int(self.cfg.mlp_topk)
        if k <= 0:
            return

        act32 = act.astype(mx.float32)
        abs_act = mx.abs(act32)
        top_abs, top_idx = _topk_with_indices(abs_act, k=k)
        top_val = mx.take_along_axis(act32, top_idx, axis=-1)
        mx.eval(top_idx, top_abs, top_val)
        idx_tk: List[List[int]] = top_idx[0].tolist()
        abs_tk: List[List[float]] = top_abs[0].tolist()
        val_tk: List[List[float]] = top_val[0].tolist()

        layer = self.layers[int(layer_id)]
        n = int(start_pos) + len(idx_tk)
        _ensure_len(layer.mlp_topk, n, fill=None)
        for i in range(len(idx_tk)):
            layer.mlp_topk[int(start_pos) + i] = {
                "idx": [int(x) for x in idx_tk[i]],
                "abs": [float(x) for x in abs_tk[i]],
                "val": [float(x) for x in val_tk[i]],
            }

    def record_attn(
        self,
        *,
        layer_id: int,
        start_pos: int,
        attn: mx.array,
        query_positions: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Record attention weights as top-k keys per head.

        Args:
          attn: [B, heads, Q, K] (B=1). Q tokens correspond to absolute positions:
                `query_positions` if provided, else `range(start_pos, start_pos+Q)`.
        """
        if not self.cfg.record_attn:
            return
        k = int(self.cfg.attn_topk)
        if k <= 0:
            return

        bsz, heads, q_len, k_len = attn.shape
        if int(bsz) != 1:
            raise ValueError("record_attn currently supports batch_size=1 only")

        if self.num_heads is None:
            self.num_heads = int(heads)

        if query_positions is None:
            query_positions = list(range(int(start_pos), int(start_pos) + int(q_len)))
        if len(query_positions) != int(q_len):
            raise ValueError("query_positions length mismatch")

        attn32 = attn.astype(mx.float32)
        top_w, top_idx = _topk_with_indices(attn32, k=k)  # [1, heads, Q, k]
        mx.eval(top_w, top_idx)
        top_w_l: List[List[List[float]]] = top_w[0].tolist()
        top_i_l: List[List[List[int]]] = top_idx[0].tolist()

        layer = self.layers[int(layer_id)]
        for qi, qpos in enumerate(query_positions):
            _ensure_len(layer.attn_topk, int(qpos) + 1, fill=None)
            per_head: List[Dict[str, Any]] = []
            for h in range(int(heads)):
                per_head.append(
                    {
                        "idx": [int(x) for x in top_i_l[h][qi]],
                        "w": [float(x) for x in top_w_l[h][qi]],
                        "k_len": int(k_len),
                    }
                )
            layer.attn_topk[int(qpos)] = per_head

    def to_dict(self, *, tokenizer: Any = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        token_ids = [int(t) if t is not None else -1 for t in self.token_ids]
        token_strs: Optional[List[str]] = None
        token_texts: Optional[List[Optional[str]]] = None
        if tokenizer is not None:
            try:
                token_strs = [str(t) for t in tokenizer.convert_ids_to_tokens(token_ids)]
            except Exception:
                token_strs = None
            try:
                token_texts = []
                for tid in token_ids:
                    if int(tid) < 0:
                        token_texts.append(None)
                        continue
                    token_texts.append(
                        str(
                            tokenizer.decode(
                                [int(tid)],
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False,
                            )
                        )
                    )
            except Exception:
                token_texts = None

        metrics: Dict[str, List[List[Optional[float]]]] = {}
        for layer_id, layer in enumerate(self.layers):
            for name, series in layer.metrics.items():
                mat = metrics.get(name)
                if mat is None:
                    mat = []
                    metrics[name] = mat
                mat.append(list(series))

        out: Dict[str, Any] = {
            "meta": {
                "created_at_s": float(self.created_at_s),
                "num_layers": int(self.num_layers),
                "num_tokens": int(self.num_tokens),
                "num_heads": int(self.num_heads) if self.num_heads is not None else None,
                **(meta or {}),
            },
            "trace_cfg": {
                "record_qkv": bool(self.cfg.record_qkv),
                "record_qkv_per_head": bool(self.cfg.record_qkv_per_head),
                "mlp_topk": int(self.cfg.mlp_topk),
                "record_attn": bool(self.cfg.record_attn),
                "record_attn_all_queries": bool(self.cfg.record_attn_all_queries),
                "attn_topk": int(self.cfg.attn_topk),
            },
            "tokens": [
                {
                    "pos": i,
                    "id": tid,
                    "tok": (token_strs[i] if token_strs else None),
                    "text": (token_texts[i] if token_texts else None),
                }
                for i, tid in enumerate(token_ids)
            ],
            "metrics": metrics,
            "mlp_topk": [layer.mlp_topk for layer in self.layers] if int(self.cfg.mlp_topk) > 0 else None,
            "qkv_rms_by_head": [
                layer.qkv_rms_by_head for layer in self.layers
            ]
            if bool(self.cfg.record_qkv and self.cfg.record_qkv_per_head)
            else None,
            "attn_topk": [layer.attn_topk for layer in self.layers] if bool(self.cfg.record_attn) else None,
        }
        return out


def _flatten_arrays(obj: Any) -> List[mx.array]:
    if isinstance(obj, mx.array):
        return [obj]
    if isinstance(obj, dict):
        out: List[mx.array] = []
        for v in obj.values():
            out.extend(_flatten_arrays(v))
        return out
    if isinstance(obj, (list, tuple)):
        out = []
        for v in obj:
            out.extend(_flatten_arrays(v))
        return out
    return []


class TimingTracer:
    def __init__(self, *, record_memory: bool = False) -> None:
        self.created_at_s = time.time()
        self.record_memory = bool(record_memory)
        self.events: List[Dict[str, Any]] = []
        self.prefix = ""

    def set_prefix(self, prefix: Optional[str]) -> None:
        self.prefix = str(prefix) if prefix else ""

    def start(self) -> float:
        return time.perf_counter()

    def end(
        self,
        name: str,
        t0: float,
        *,
        arrays: Optional[Sequence[Any]] = None,
        depth: int = 0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> float:
        if arrays:
            flat = []
            for item in arrays:
                flat.extend(_flatten_arrays(item))
            if flat:
                mx.eval(*flat)
        ms = (time.perf_counter() - float(t0)) * 1000.0
        event: Dict[str, Any] = {
            "name": f"{self.prefix}/{name}" if self.prefix else str(name),
            "ms": float(ms),
            "depth": int(depth),
        }
        if meta:
            event["meta"] = meta
        if self.record_memory:
            try:
                event["active_mem_mib"] = float(mx.get_active_memory()) / 1024.0 / 1024.0
                event["peak_mem_mib"] = float(mx.get_peak_memory()) / 1024.0 / 1024.0
            except Exception:
                pass
        self.events.append(event)
        return ms

    def summary(self, *, top_n: int = 12) -> List[Tuple[str, float]]:
        totals: Dict[str, float] = {}
        for event in self.events:
            name = str(event.get("name", ""))
            totals[name] = totals.get(name, 0.0) + float(event.get("ms", 0.0))
        pairs = sorted(totals.items(), key=lambda x: x[1], reverse=True)
        return pairs[: max(0, int(top_n))]

    def to_dict(self, *, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "meta": {
                "created_at_s": float(self.created_at_s),
                "record_memory": bool(self.record_memory),
                **(meta or {}),
            },
            "events": list(self.events),
        }


def render_trace_html(trace: Dict[str, Any]) -> str:
    # Minimal self-contained HTML report (no external deps).
    payload = json.dumps(trace, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>MLX Activation Trace</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 16px; }}
    .row {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
    select, input {{ padding: 6px 8px; font-size: 14px; }}
    #wrap {{ position: relative; }}
    #tip {{
      position: absolute; pointer-events: none; display: none; z-index: 10;
      background: rgba(0,0,0,0.85); color: #fff; padding: 8px 10px; border-radius: 8px;
      font-size: 12px; max-width: 420px; white-space: pre-wrap;
    }}
    canvas {{ border: 1px solid #ddd; border-radius: 10px; }}
    .muted {{ color: #666; font-size: 12px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; margin-top: 14px; }}
    .card {{ border: 1px solid #e5e5e5; border-radius: 12px; padding: 12px; }}
    .card h3 {{ margin: 0 0 10px 0; font-size: 14px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
  </style>
</head>
<body>
  <div class="grid">
    <div class="card">
      <h3>Activation RMS Heatmap</h3>
      <div class="row">
        <div><strong>Metric</strong> <select id="metric"></select></div>
        <div><strong>Cell</strong> <span class="muted">hover for details</span></div>
      </div>
      <div id="wrap">
        <canvas id="heat" width="1200" height="720"></canvas>
        <div id="tip"></div>
      </div>
      <div class="muted" style="margin-top:10px;">
        Y axis: layer (0 at top). X axis: token position. Color: normalized value (per metric).
      </div>
    </div>

    <div class="card" id="attn-card" style="display:none;">
      <h3>Attention Pattern (Top-k Keys per Head)</h3>
      <div class="row">
        <div><strong>Layer</strong> <select id="attnLayer"></select></div>
        <div><strong>Head</strong> <select id="attnHead"></select></div>
        <div class="muted">X: query token pos, Y: key token pos (0 at top)</div>
      </div>
      <canvas id="attn" width="1200" height="520"></canvas>
      <div class="muted" id="attn-note" style="margin-top:10px;"></div>
      <pre class="mono" id="attn-detail" style="margin:10px 0 0 0; background:#fafafa; padding:10px; border-radius:10px; overflow:auto;"></pre>
    </div>
  </div>

  <script id="trace-data" type="application/json">{payload}</script>
  <script>
    const trace = JSON.parse(document.getElementById('trace-data').textContent);
    const metrics = trace.metrics || {{}};
    const metricNames = Object.keys(metrics).sort();
    const metricSel = document.getElementById('metric');
    for (const name of metricNames) {{
      const opt = document.createElement('option');
      opt.value = name; opt.textContent = name;
      metricSel.appendChild(opt);
    }}
    if (metricNames.length === 0) {{
      metricSel.innerHTML = '<option value="">(no metrics)</option>';
    }}

    const canvas = document.getElementById('heat');
    const ctx = canvas.getContext('2d');
    const tip = document.getElementById('tip');
    const tokens = trace.tokens || [];
    function clamp01(x) {{ return Math.max(0, Math.min(1, x)); }}
    function color(t) {{
      // simple blue->cyan->yellow->red
      t = clamp01(t);
      const a = clamp01(t/0.33), b = clamp01((t-0.33)/0.33), c = clamp01((t-0.66)/0.34);
      let r=0,g=0,bl=0;
      if (t < 0.33) {{ r = 0; g = Math.round(255*a); bl = 255; }}
      else if (t < 0.66) {{ r = Math.round(255*b); g = 255; bl = Math.round(255*(1-b)); }}
      else {{ r = 255; g = Math.round(255*(1-c)); bl = 0; }}
      return `rgb(${{r}},${{g}},${{bl}})`;
    }}

    function flatten(mat) {{
      const out = [];
      for (const row of mat) for (const v of row) if (v !== null && v !== undefined) out.push(v);
      return out;
    }}

    let current = metricNames[0] || '';
    function draw() {{
      current = metricSel.value;
      ctx.clearRect(0,0,canvas.width,canvas.height);
      const mat = metrics[current] || [];
      const L = mat.length || 0;
      const T = (mat[0] || []).length || 0;
      if (L === 0 || T === 0) return;

      const vals = flatten(mat);
      let vmin = Infinity, vmax = -Infinity;
      for (const v of vals) {{ if (v < vmin) vmin = v; if (v > vmax) vmax = v; }}
      if (!isFinite(vmin) || !isFinite(vmax) || vmin === vmax) {{ vmin = 0; vmax = 1; }}

      const pad = 12;
      const w = canvas.width - pad*2;
      const h = canvas.height - pad*2;
      const cw = w / T;
      const ch = h / L;
      for (let y=0; y<L; y++) {{
        for (let x=0; x<T; x++) {{
          const v = mat[y][x];
          const t = (v === null || v === undefined) ? 0 : (v - vmin) / (vmax - vmin);
          ctx.fillStyle = (v === null || v === undefined) ? 'rgb(245,245,245)' : color(t);
          ctx.fillRect(pad + x*cw, pad + y*ch, Math.ceil(cw), Math.ceil(ch));
        }}
      }}
      ctx.strokeStyle = '#ddd';
      ctx.strokeRect(pad, pad, w, h);
    }}

    function hitTest(ev) {{
      const mat = metrics[current] || [];
      const L = mat.length || 0;
      const T = (mat[0] || []).length || 0;
      if (L === 0 || T === 0) return null;
      const pad = 12;
      const rect = canvas.getBoundingClientRect();
      const x = ev.clientX - rect.left;
      const y = ev.clientY - rect.top;
      const w = canvas.width - pad*2;
      const h = canvas.height - pad*2;
      if (x < pad || y < pad || x > pad+w || y > pad+h) return null;
      const tx = Math.floor((x - pad) / (w / T));
      const ly = Math.floor((y - pad) / (h / L));
      if (tx < 0 || ly < 0 || tx >= T || ly >= L) return null;
      return {{ tokenPos: tx, layer: ly, value: mat[ly][tx] }};
    }}

    canvas.addEventListener('mousemove', (ev) => {{
      const hit = hitTest(ev);
      if (!hit) {{ tip.style.display = 'none'; return; }}
      const tok = tokens[hit.tokenPos] || {{}};
      const tokLabel = (tok.text ?? tok.tok ?? '');
      const tokId = tok.id ?? -1;
      const v = hit.value;
      const text = `metric: ${{current}}\\nlayer: ${{hit.layer}}\\npos: ${{hit.tokenPos}}\\ntoken_id: ${{tokId}}\\ntoken: ${{tokLabel}}\\nvalue: ${{v}}`;
      tip.textContent = text;
      tip.style.left = (ev.offsetX + 16) + 'px';
      tip.style.top = (ev.offsetY + 16) + 'px';
      tip.style.display = 'block';
    }});
    canvas.addEventListener('mouseleave', () => tip.style.display = 'none');
    metricSel.addEventListener('change', draw);
    draw();

    // --- Attention top-k visualization (scatter plot) ---
    const attnTopk = trace.attn_topk || null; // [layer][pos] -> [head] -> {{idx,w,k_len}}
    const attnCard = document.getElementById('attn-card');
    const attnCanvas = document.getElementById('attn');
    const attnCtx = attnCanvas.getContext('2d');
    const attnLayerSel = document.getElementById('attnLayer');
    const attnHeadSel = document.getElementById('attnHead');
    const attnNote = document.getElementById('attn-note');
    const attnDetail = document.getElementById('attn-detail');

    function tokLabel(pos) {{
      const t = tokens[pos] || {{}};
      const sTok = (t.tok === null || t.tok === undefined) ? '' : String(t.tok);
      const sText = (t.text === null || t.text === undefined) ? '' : String(t.text);
      const show = sText !== '' ? sText : sTok;
      return `${{pos}}: id=${{t.id}} text=${{JSON.stringify(show)}} tok=${{JSON.stringify(sTok)}}`;
    }}

    function initAttn() {{
      if (!attnTopk || !Array.isArray(attnTopk) || attnTopk.length === 0) return;
      attnCard.style.display = 'block';
      const numLayers = attnTopk.length;
      for (let i=0;i<numLayers;i++) {{
        const opt = document.createElement('option');
        opt.value = String(i); opt.textContent = String(i);
        attnLayerSel.appendChild(opt);
      }}
      const heads = trace.meta && trace.meta.num_heads ? trace.meta.num_heads : null;
      const numHeads = heads || 1;
      for (let h=0; h<numHeads; h++) {{
        const opt = document.createElement('option');
        opt.value = String(h); opt.textContent = String(h);
        attnHeadSel.appendChild(opt);
      }}
      attnNote.textContent =
        (trace.trace_cfg && trace.trace_cfg.record_attn_all_queries)
          ? 'Includes prompt-token attention (prefill) + decode tokens.'
          : 'By default only records the last query in each forward (last prompt token + generated tokens). Use --trace_attn_all_queries to include all prompt tokens.';
      attnLayerSel.addEventListener('change', drawAttn);
      attnHeadSel.addEventListener('change', drawAttn);
      attnCanvas.addEventListener('mousemove', onAttnMove);
      attnCanvas.addEventListener('mouseleave', () => attnDetail.textContent = '');
      drawAttn();
    }}

    function drawAttn() {{
      const layer = parseInt(attnLayerSel.value || '0', 10);
      const head = parseInt(attnHeadSel.value || '0', 10);
      const T = tokens.length;
      attnCtx.clearRect(0,0,attnCanvas.width, attnCanvas.height);
      if (!attnTopk || !attnTopk[layer]) return;

      const pad = 12;
      const w = attnCanvas.width - pad*2;
      const h = attnCanvas.height - pad*2;
      const cw = w / Math.max(1, T);
      const ch = h / Math.max(1, T);

      // background + diagonal
      attnCtx.fillStyle = '#fff';
      attnCtx.fillRect(0,0,attnCanvas.width, attnCanvas.height);
      attnCtx.strokeStyle = '#eee';
      attnCtx.strokeRect(pad, pad, w, h);
      attnCtx.beginPath();
      attnCtx.moveTo(pad, pad);
      attnCtx.lineTo(pad + w, pad + h);
      attnCtx.strokeStyle = '#f0f0f0';
      attnCtx.stroke();

      // collect max weight for scaling
      let maxW = 0;
      for (let q=0; q<T; q++) {{
        const entry = attnTopk[layer][q];
        if (!entry || !entry[head]) continue;
        const ws = entry[head].w || [];
        for (const v of ws) maxW = Math.max(maxW, v);
      }}
      if (maxW <= 0) maxW = 1;

      for (let q=0; q<T; q++) {{
        const entry = attnTopk[layer][q];
        if (!entry || !entry[head]) continue;
        const idx = entry[head].idx || [];
        const ws = entry[head].w || [];
        for (let i=0; i<idx.length; i++) {{
          const kpos = idx[i];
          const weight = ws[i] || 0;
          const alpha = Math.max(0.05, Math.min(1.0, weight / maxW));
          const x = pad + (q + 0.5) * cw;
          const y = pad + (kpos + 0.5) * ch;
          attnCtx.fillStyle = `rgba(30, 30, 30, ${{alpha}})`;
          attnCtx.fillRect(x-1, y-1, 2, 2);
        }}
      }}
    }}

    function onAttnMove(ev) {{
      const T = tokens.length;
      const pad = 12;
      const rect = attnCanvas.getBoundingClientRect();
      const x = ev.clientX - rect.left;
      if (x < pad || x > attnCanvas.width - pad) return;
      const w = attnCanvas.width - pad*2;
      const q = Math.max(0, Math.min(T-1, Math.floor((x - pad) / (w / Math.max(1, T)))));

      const layer = parseInt(attnLayerSel.value || '0', 10);
      const head = parseInt(attnHeadSel.value || '0', 10);
      const entry = (attnTopk && attnTopk[layer]) ? attnTopk[layer][q] : null;
      if (!entry || !entry[head]) {{
        attnDetail.textContent = `query: ${{tokLabel(q)}}\\n(no attention recorded here)`;
        return;
      }}
      const idx = entry[head].idx || [];
      const ws = entry[head].w || [];
      const lines = [];
      lines.push(`layer=${{layer}} head=${{head}}`);
      lines.push(`query: ${{tokLabel(q)}}`);
      lines.push(`topk keys:`);
      for (let i=0; i<idx.length; i++) {{
        const kpos = idx[i];
        const wv = ws[i];
        lines.push(`  w=${{wv.toFixed(6)}}  key: ${{tokLabel(kpos)}}`);
      }}
      attnDetail.textContent = lines.join('\\n');
    }}

    initAttn();
  </script>
</body>
</html>"""


def write_trace_outputs(*, out_path: str | Path, trace: Dict[str, Any]) -> Tuple[Path, Path]:
    out_path = Path(out_path)
    if out_path.suffix.lower() == ".json":
        json_path = out_path
        html_path = out_path.with_suffix(".html")
        out_dir = json_path.parent
    else:
        out_dir = out_path
        json_path = out_dir / "trace.json"
        html_path = out_dir / "trace.html"

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_trace_html(trace), encoding="utf-8")
    return json_path, html_path


def write_timing_trace(*, out_path: str | Path, trace: Dict[str, Any]) -> Path:
    out = Path(out_path)
    if out.suffix.lower() == ".json":
        json_path = out
    else:
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "timing.json"
    json_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path
