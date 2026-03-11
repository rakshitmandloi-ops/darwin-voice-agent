"""
Interactive evolution dashboard with zoomable graph.
python dashboard.py -> http://localhost:8050
"""

import json
import difflib
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

LOGS_DIR = Path(__file__).parent / "logs"

HTML = r"""<!DOCTYPE html>
<html><head>
<title>Darwin Evolution</title>
<meta charset="utf-8">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:monospace;background:#0d1117;color:#c9d1d9;overflow:hidden;height:100vh;display:flex;flex-direction:column}
#topbar{padding:12px 20px;background:#161b22;border-bottom:1px solid #30363d;display:flex;gap:20px;align-items:center;flex-shrink:0;z-index:10}
#topbar h1{color:#58a6ff;font-size:1.1em;white-space:nowrap}
.chip{background:#21262d;padding:4px 10px;border-radius:4px;font-size:0.8em;display:flex;gap:6px;align-items:center}
.chip .cl{font-size:0.75em;color:#8b949e}
.chip .cv{font-weight:bold}
.green{color:#3fb950}.red{color:#f85149}.yellow{color:#d29922}.blue{color:#58a6ff}.dim{color:#484f58}
#main{flex:1;position:relative;overflow:hidden}
canvas{position:absolute;top:0;left:0;cursor:grab}
canvas:active{cursor:grabbing}

/* Tooltip */
#tooltip{position:fixed;background:#1c2128;border:1px solid #444c56;border-radius:8px;padding:14px;font-size:0.8em;max-width:600px;pointer-events:auto;display:none;z-index:100;box-shadow:0 8px 24px rgba(0,0,0,0.6);line-height:1.5;max-height:80vh;overflow-y:auto}
#tooltip h3{color:#58a6ff;margin-bottom:6px;font-size:1em}
#tooltip .trow{display:flex;justify-content:space-between;padding:2px 0}
#tooltip .tk{color:#8b949e}
#tooltip .tv{color:#c9d1d9;font-weight:bold}
#tooltip .sep{border-top:1px solid #30363d;margin:6px 0}
#tooltip .tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:0.85em;margin:1px}
#tooltip .tag.pass{background:#0d2818;color:#3fb950}
#tooltip .tag.fail{background:#200d0d;color:#f85149}
#tooltip .diff-block{background:#0d1117;border:1px solid #30363d;border-radius:4px;padding:8px;margin-top:6px;max-height:300px;overflow-y:auto;font-size:0.85em;white-space:pre-wrap;word-break:break-word;line-height:1.4}
#tooltip .diff-block .dl-add{color:#3fb950;background:#0d2818}
#tooltip .diff-block .dl-del{color:#f85149;background:#200d0d}
#tooltip .diff-block .dl-hdr{color:#58a6ff;font-weight:bold}
#tooltip .diff-block .dl-ctx{color:#8b949e}
#tooltip .diff-comp{color:#d29922;font-weight:bold;margin-top:8px;margin-bottom:2px;font-size:0.95em}

/* Detail panel */
#detail{position:fixed;right:0;top:0;bottom:0;width:380px;background:#161b22;border-left:1px solid #30363d;z-index:50;display:none;overflow-y:auto;padding:20px;transform:translateX(100%);transition:transform 0.2s}
#detail.open{display:block;transform:translateX(0)}
#detail h2{color:#58a6ff;font-size:1.1em;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center}
#detail .close{cursor:pointer;color:#8b949e;font-size:1.3em;padding:4px 8px}
#detail .close:hover{color:#f85149}
#detail .section{margin-bottom:16px}
#detail .section h3{color:#8b949e;font-size:0.8em;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;padding-bottom:4px;border-bottom:1px solid #21262d}
#detail .drow{display:flex;justify-content:space-between;padding:3px 0;font-size:0.85em}
#detail .dk{color:#8b949e}
#detail .dv{font-weight:bold}
#detail .persona-bar{display:flex;align-items:center;gap:8px;padding:3px 0;font-size:0.85em}
#detail .persona-bar .bar-bg{flex:1;height:8px;background:#21262d;border-radius:4px;overflow:hidden}
#detail .persona-bar .bar-fg{height:100%;border-radius:4px}
#detail .comp-list{font-size:0.8em;color:#8b949e;margin-top:4px}
#detail .mutation-text{font-size:0.8em;color:#8b949e;font-style:italic;margin-top:6px;padding:8px;background:#0d1117;border-radius:4px;max-height:120px;overflow-y:auto}
#d_tabs{display:flex;gap:4px;margin-bottom:12px;flex-wrap:wrap}
#d_tabs button{background:#21262d;border:1px solid #30363d;color:#8b949e;padding:5px 10px;border-radius:4px;cursor:pointer;font-family:monospace;font-size:0.8em}
#d_tabs button:hover{background:#30363d;color:#c9d1d9}
#d_tabs button.active{background:#58a6ff22;border-color:#58a6ff;color:#58a6ff}
.score-grid{display:grid;grid-template-columns:auto 1fr 50px;gap:2px 8px;align-items:center;font-size:0.85em;margin:6px 0}
.score-grid .sg-label{color:#8b949e}
.score-grid .sg-bar{height:10px;background:#21262d;border-radius:3px;overflow:hidden}
.score-grid .sg-fill{height:100%;border-radius:3px}
.score-grid .sg-val{text-align:right;font-weight:bold}
.stage-header{color:#58a6ff;font-weight:bold;font-size:0.9em;margin:12px 0 6px;padding-bottom:4px;border-bottom:1px solid #21262d}
.rule-row{display:flex;justify-content:space-between;padding:2px 0;font-size:0.82em}
.rule-row .rr-icon{width:18px}
.diff-section{margin-top:12px}
.diff-section .diff-comp{color:#d29922;font-weight:bold;font-size:0.9em;margin-top:10px;margin-bottom:4px}
.diff-section .diff-block{background:#0d1117;border:1px solid #30363d;border-radius:4px;padding:8px;max-height:250px;overflow-y:auto;font-size:0.8em;white-space:pre-wrap;word-break:break-word;line-height:1.4}
.diff-section .dl-add{color:#3fb950;background:#0d2818}
.diff-section .dl-del{color:#f85149;background:#200d0d}
.diff-section .dl-hdr{color:#58a6ff;font-weight:bold}
.diff-section .dl-ctx{color:#6e7681}

/* Zoom controls */
#controls{position:fixed;bottom:16px;right:400px;z-index:20;display:flex;gap:6px}
#controls button{background:#21262d;border:1px solid #30363d;color:#c9d1d9;width:32px;height:32px;border-radius:6px;cursor:pointer;font-size:1.1em;display:flex;align-items:center;justify-content:center}
#controls button:hover{background:#30363d}
</style>
</head>
<body>

<div id="topbar">
  <h1>DARWIN EVOLUTION</h1>
  <div class="chip"><span class="cl">Gen</span><span class="cv" id="s_gen">-</span></div>
  <div class="chip"><span class="cl">Best</span><span class="cv green" id="s_best">-</span></div>
  <div class="chip"><span class="cl">Seed</span><span class="cv" id="s_seed">-</span></div>
  <div class="chip"><span class="cl">Improv</span><span class="cv" id="s_imp">-</span></div>
  <div class="chip"><span class="cl">Variants</span><span class="cv" id="s_vars">-</span></div>
  <div class="chip"><span class="cl">Spent</span><span class="cv yellow" id="s_cost">-</span></div>
  <div class="chip"><span class="cl">Calls</span><span class="cv dim" id="s_calls">-</span></div>
  <select id="batch_select" onchange="switchBatch(this.value)" style="background:#21262d;color:#c9d1d9;border:1px solid #30363d;padding:4px 8px;border-radius:4px;font-family:monospace;font-size:0.8em">
    <option value="">Loading batches...</option>
  </select>
</div>

<div id="main">
  <canvas id="c"></canvas>
</div>

<div id="tooltip"></div>

<div id="detail">
  <h2><span id="d_title">Node</span><span class="close" onclick="closeDetail()">&times;</span></h2>
  <div id="d_tabs"></div>
  <div id="d_body"></div>
</div>

<div id="controls">
  <button onclick="zoomIn()">+</button>
  <button onclick="zoomOut()">&minus;</button>
  <button onclick="resetView()">&#8634;</button>
</div>

<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const detail = document.getElementById('detail');

let DATA = null, NODES = [], EDGES = [], POS = {};
let zoom = 1, panX = 0, panY = 0;
let dragging = false, dragStartX, dragStartY, dragPanX, dragPanY;
let hoveredEdge = null, selectedNode = null;
let currentBatch = null;  // null = current/latest

const NW = 180, NH = 70, HGAP = 30, VGAP = 110;

// --- Data ---
async function load() {
  const batchParam = currentBatch ? '&batch='+currentBatch : '';
  const r = await fetch('/api/state?t='+Date.now()+batchParam);
  DATA = await r.json();
  updateTopbar();
  layoutTree();
  draw();
  loadBatches();
}

async function loadBatches() {
  try {
    const r = await fetch('/api/batches');
    const batches = await r.json();
    const sel = document.getElementById('batch_select');
    const curVal = sel.value;
    sel.innerHTML = '';
    batches.forEach(b => {
      const opt = document.createElement('option');
      opt.value = b.batch_id;
      const status = b.status === 'complete' ? ' [done]' : ' [live]';
      opt.textContent = b.batch_id + status + ' (' + (b.variants||0) + ' variants)';
      if (b.batch_id === currentBatch || (!currentBatch && batches.indexOf(b) === 0)) opt.selected = true;
      sel.appendChild(opt);
    });
  } catch(e) {}
}

function switchBatch(batchId) {
  currentBatch = batchId;
  diffCache = {};
  load();
}

function updateTopbar() {
  const s = DATA.summary;
  const imp = s.best_score - s.seed_score;
  document.getElementById('s_gen').textContent = s.max_generation;
  document.getElementById('s_best').textContent = s.best_score.toFixed(2);
  document.getElementById('s_seed').textContent = s.seed_score.toFixed(2);
  const impEl = document.getElementById('s_imp');
  impEl.textContent = (imp>=0?'+':'')+imp.toFixed(2);
  impEl.className = 'cv ' + (imp>0.05?'green':imp<-0.05?'red':'yellow');
  document.getElementById('s_vars').textContent = s.active + '/' + s.total_variants;
  document.getElementById('s_cost').textContent = '$' + DATA.budget.spent.toFixed(2);
  document.getElementById('s_calls').textContent = DATA.budget.total_calls.toLocaleString();
}

function layoutTree() {
  NODES = DATA.variants;
  const byId = {};
  NODES.forEach(n => byId[n.version_id] = n);

  const kids = {};
  NODES.forEach(n => {
    if (n.parent_id) {
      if (!kids[n.parent_id]) kids[n.parent_id] = [];
      kids[n.parent_id].push(n.version_id);
    }
  });

  POS = {};
  let cx = 40;

  function lay(id, depth) {
    const ch = kids[id] || [];
    if (!ch.length) {
      POS[id] = {x: cx, y: 50 + depth*(NH+VGAP)};
      cx += NW + HGAP;
      return;
    }
    ch.forEach(c => lay(c, depth+1));
    const cp = ch.map(c => POS[c]);
    const l = Math.min(...cp.map(p=>p.x));
    const r = Math.max(...cp.map(p=>p.x));
    POS[id] = {x: (l+r)/2, y: 50 + depth*(NH+VGAP)};
  }

  NODES.filter(n => !n.parent_id).forEach(n => lay(n.version_id, 0));
  NODES.forEach(n => {
    if (!POS[n.version_id]) {
      POS[n.version_id] = {x: cx, y: 50 + n.generation*(NH+VGAP)};
      cx += NW + HGAP;
    }
  });

  // Build edges
  EDGES = [];
  NODES.forEach(n => {
    if (n.parent_id && POS[n.parent_id]) {
      const p = byId[n.parent_id];
      EDGES.push({from: n.parent_id, to: n.version_id, child: n, parent: p});
    }
  });
}

// --- Drawing ---
function resize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight - document.getElementById('topbar').offsetHeight;
}

function toScreen(x, y) {
  return [(x + panX) * zoom, (y + panY) * zoom];
}

function toWorld(sx, sy) {
  return [sx/zoom - panX, sy/zoom - panY];
}

function draw() {
  resize();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.scale(zoom, zoom);
  ctx.translate(panX, panY);

  const bestId = DATA.summary.best_id;

  // Edges
  EDGES.forEach(e => {
    const fp = POS[e.from], tp = POS[e.to];
    if (!fp || !tp) return;
    const x1 = fp.x+NW/2, y1 = fp.y+NH;
    const x2 = tp.x+NW/2, y2 = tp.y;
    const diff = e.child.avg_score - (e.parent ? e.parent.avg_score : 0);
    const col = diff>0.15?'#3fb950':diff<-0.15?'#f85149':'#484f58';
    const isHov = hoveredEdge === e;

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.bezierCurveTo(x1, y1+(y2-y1)*0.5, x2, y1+(y2-y1)*0.5, x2, y2);
    ctx.strokeStyle = isHov ? '#58a6ff' : col;
    ctx.lineWidth = isHov ? 3.5 : 2;
    ctx.setLineDash(e.child.discarded ? [6,4] : []);
    ctx.stroke();
    ctx.setLineDash([]);

    // Arrow
    ctx.beginPath();
    ctx.moveTo(x2, y2); ctx.lineTo(x2-5, y2-8); ctx.lineTo(x2+5, y2-8); ctx.closePath();
    ctx.fillStyle = isHov ? '#58a6ff' : col;
    ctx.fill();

    // Edge label
    const mx = (x1+x2)/2, my = (y1+y2)/2;
    const diffStr = (diff>=0?'+':'')+diff.toFixed(2);
    const comps = e.child.components_modified.map(c=>c.replace('_prompt','')).join('+');
    const label = comps ? comps + ' ' + diffStr : diffStr;

    ctx.font = 'bold 9px monospace';
    const tw = ctx.measureText(label).width + 12;
    ctx.fillStyle = '#0d1117';
    ctx.beginPath();
    ctx.roundRect(mx-tw/2, my-10, tw, 18, 4);
    ctx.fill();
    ctx.strokeStyle = isHov ? '#58a6ff55' : col+'44';
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = isHov ? '#58a6ff' : col;
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, mx, my);
  });

  // Nodes
  NODES.forEach(n => {
    const p = POS[n.version_id];
    if (!p) return;
    const isBest = n.version_id === bestId;
    const isSeed = !n.parent_id;
    const isSel = selectedNode === n.version_id;

    ctx.globalAlpha = n.discarded ? 0.35 : 1;

    let bg, border;
    if (isBest) {bg='#0d2818';border='#3fb950';}
    else if (n.discarded) {bg='#200d0d';border='#f85149';}
    else if (n.promoted) {bg='#1c1a0d';border='#d29922';}
    else if (isSeed) {bg='#0d1a28';border='#58a6ff';}
    else {bg='#161b22';border='#30363d';}

    if (isSel) {
      ctx.shadowColor = '#58a6ff'; ctx.shadowBlur = 16;
    } else if (isBest) {
      ctx.shadowColor = '#3fb950'; ctx.shadowBlur = 10;
    }

    ctx.fillStyle = bg;
    ctx.strokeStyle = border;
    ctx.lineWidth = isSel ? 3 : isBest ? 2.5 : 1.5;
    ctx.beginPath();
    ctx.roundRect(p.x, p.y, NW, NH, 8);
    ctx.fill(); ctx.stroke();
    ctx.shadowBlur = 0;

    // Score
    ctx.fillStyle = border;
    ctx.font = 'bold 20px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(n.avg_score.toFixed(2), p.x+10, p.y+8);

    // Convos
    ctx.fillStyle = '#8b949e';
    ctx.font = '9px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(n.num_convos+'c', p.x+NW-10, p.y+12);

    // ID
    ctx.textAlign = 'left';
    ctx.fillStyle = '#6e7681';
    ctx.font = '9px monospace';
    ctx.fillText(n.version_id.slice(0,14), p.x+10, p.y+36);

    // Badge
    let badge='', bc='';
    if (isBest) {badge='\u25c6 BEST';bc='#3fb950';}
    else if (n.discarded) {badge='\u2717 DISC';bc='#f85149';}
    else if (n.promoted) {badge='\u2605 PROM';bc='#d29922';}
    else if (isSeed) {badge='\u25cf SEED';bc='#58a6ff';}
    if (badge) {
      ctx.fillStyle = bc;
      ctx.font = 'bold 9px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(badge, p.x+10, p.y+52);
    }

    ctx.fillStyle = '#484f58';
    ctx.font = '9px monospace';
    ctx.textAlign = 'right';
    ctx.fillText('gen'+n.generation, p.x+NW-10, p.y+52);

    ctx.globalAlpha = 1;
  });

  ctx.restore();
}

// --- Interaction ---
function hitTestEdge(wx, wy) {
  for (const e of EDGES) {
    const fp = POS[e.from], tp = POS[e.to];
    if (!fp||!tp) continue;
    const x1=fp.x+NW/2, y1=fp.y+NH, x2=tp.x+NW/2, y2=tp.y;
    const mx=(x1+x2)/2, my=(y1+y2)/2;
    if (Math.abs(wx-mx)<40 && Math.abs(wy-my)<20) return e;
  }
  return null;
}

function hitTestNode(wx, wy) {
  for (const n of NODES) {
    const p = POS[n.version_id];
    if (!p) continue;
    if (wx>=p.x && wx<=p.x+NW && wy>=p.y && wy<=p.y+NH) return n;
  }
  return null;
}

canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const [wx, wy] = toWorld(mx, my);
  const dz = e.deltaY > 0 ? 0.9 : 1.1;
  zoom *= dz;
  zoom = Math.max(0.2, Math.min(3, zoom));
  panX = mx/zoom - wx;
  panY = my/zoom - wy;
  draw();
}, {passive: false});

canvas.addEventListener('mousedown', e => {
  dragging = true;
  dragStartX = e.clientX; dragStartY = e.clientY;
  dragPanX = panX; dragPanY = panY;
});

canvas.addEventListener('mousemove', e => {
  if (dragging) {
    panX = dragPanX + (e.clientX - dragStartX)/zoom;
    panY = dragPanY + (e.clientY - dragStartY)/zoom;
    draw();
    return;
  }

  const rect = canvas.getBoundingClientRect();
  const [wx, wy] = toWorld(e.clientX - rect.left, e.clientY - rect.top);

  // Edge hover
  const edge = hitTestEdge(wx, wy);
  if (edge !== hoveredEdge) {
    hoveredEdge = edge;
    draw();
  }
  if (edge) {
    // Quick preview tooltip on hover
    const c = edge.child, p = edge.parent;
    const diff = c.avg_score - (p ? p.avg_score : 0);
    const comps = c.components_modified.map(x=>x.replace('_prompt','')).join('+');
    tooltip.innerHTML = `<b>${edge.from} \u2192 ${edge.to}</b><br>${comps} <span style="color:${diff>0?'#3fb950':'#f85149'}">${diff>=0?'+':''}${diff.toFixed(2)}</span><br><span class="dim">Click for full diff</span>`;
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX+12)+'px';
    tooltip.style.top = (e.clientY+12)+'px';
    canvas.style.cursor = 'pointer';
  } else {
    tooltip.style.display = 'none';
    const node = hitTestNode(wx, wy);
    canvas.style.cursor = node ? 'pointer' : 'grab';
  }
});

canvas.addEventListener('mouseup', e => {
  if (dragging && Math.abs(e.clientX-dragStartX)<3 && Math.abs(e.clientY-dragStartY)<3) {
    // Click (not drag)
    const rect = canvas.getBoundingClientRect();
    const [wx, wy] = toWorld(e.clientX-rect.left, e.clientY-rect.top);

    // Check edge click first
    const edge = hitTestEdge(wx, wy);
    if (edge) {
      selectedNode = null;
      showEdgeDetail(edge);
      draw();
      dragging = false;
      return;
    }

    const node = hitTestNode(wx, wy);
    if (node) {
      selectedNode = node.version_id;
      showNodeDetail(node);
      draw();
    } else {
      closeDetail();
    }
  }
  dragging = false;
});

canvas.addEventListener('mouseleave', () => {
  dragging = false;
  hoveredEdge = null;
  tooltip.style.display = 'none';
  draw();
});

let diffCache = {};

async function showEdgeTooltip(edge, mx, my) {
  const c = edge.child;
  const p = edge.parent;
  const diff = c.avg_score - (p ? p.avg_score : 0);
  const dc = diff>0?'green':diff<0?'red':'';

  let h = `<h3>${edge.from} &rarr; ${edge.to}</h3>`;
  h += trow('Score change', (diff>=0?'+':'')+diff.toFixed(2), dc);
  h += trow('Parent score', p ? p.avg_score.toFixed(2) : '?');
  h += trow('Child score', c.avg_score.toFixed(2));
  h += trow('Conversations', c.num_convos);
  h += `<div class="sep"></div>`;
  h += `<div class="tk" style="margin-bottom:4px">Components modified:</div>`;
  c.components_modified.forEach(comp => {
    h += `<span class="tag pass">${comp}</span> `;
  });

  if (c.discarded) {
    h += `<div class="sep"></div>`;
    h += `<span class="tag fail">DISCARDED - compliance failure</span>`;
  }

  h += `<div class="sep"></div>`;
  h += `<div class="tk">Mutation rationale:</div>`;
  h += `<div style="color:#c9d1d9;margin-top:4px;font-size:0.95em">${c.mutation_desc || 'N/A'}</div>`;

  // Fetch and show diff
  const cacheKey = edge.from + '|' + edge.to;
  if (!diffCache[cacheKey]) {
    h += `<div class="sep"></div><div class="dim">Loading diff...</div>`;
    tooltip.innerHTML = h;
    positionTooltip(mx, my);

    try {
      const r = await fetch(`/api/diff?parent=${encodeURIComponent(edge.from)}&child=${encodeURIComponent(edge.to)}`);
      diffCache[cacheKey] = await r.json();
    } catch(e) {
      diffCache[cacheKey] = {diffs: {}, error: e.message};
    }
  }

  const dd = diffCache[cacheKey];
  if (dd.diffs && Object.keys(dd.diffs).length) {
    h += `<div class="sep"></div>`;
    h += `<div class="tk" style="margin-bottom:4px">Prompt Diff:</div>`;
    for (const [comp, info] of Object.entries(dd.diffs)) {
      const name = comp.replace('_prompt', '');
      h += `<div class="diff-comp">${name} (${info.old_len} &rarr; ${info.new_len} chars)</div>`;
      h += `<div class="diff-block">`;
      const lines = info.diff.split('\n');
      lines.forEach(line => {
        const esc = line.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        if (line.startsWith('+++') || line.startsWith('---')) {
          h += `<div class="dl-hdr">${esc}</div>`;
        } else if (line.startsWith('@@')) {
          h += `<div class="dl-hdr">${esc}</div>`;
        } else if (line.startsWith('+')) {
          h += `<div class="dl-add">${esc}</div>`;
        } else if (line.startsWith('-')) {
          h += `<div class="dl-del">${esc}</div>`;
        } else {
          h += `<div class="dl-ctx">${esc}</div>`;
        }
      });
      h += `</div>`;
    }
  }

  tooltip.innerHTML = h;
  positionTooltip(mx, my);
}

function positionTooltip(mx, my) {
  tooltip.style.display = 'block';
  tooltip.style.left = Math.min(mx + 12, window.innerWidth - 440) + 'px';
  tooltip.style.top = Math.min(my + 12, window.innerHeight - tooltip.offsetHeight - 20) + 'px';
}

const RULE_NAMES = {
  'r1_identity_disclosure': 'Identity Disclosure',
  'r2_no_false_threats': 'No False Threats',
  'r3_no_harassment': 'No Harassment',
  'r4_no_misleading_terms': 'No Misleading Terms',
  'r5_sensitive_situations': 'Sensitive Situations',
  'r6_recording_disclosure': 'Recording Disclosure',
  'r7_professional_composure': 'Professional Composure',
  'r8_data_privacy': 'Data Privacy',
};

let currentDetailNode = null;

function showNodeDetail(node) {
  currentDetailNode = node;
  tooltip.style.display = 'none';

  document.getElementById('d_title').textContent = node.version_id;

  // Build persona tabs
  const personas = [...new Set((node.convo_details||[]).map(c => c.persona))];
  let tabs = '<button class="active" onclick="showNodeTab(\'overview\')">Overview</button>';
  personas.forEach(p => {
    tabs += `<button onclick="showNodeTab('${p}')">${p}</button>`;
  });
  tabs += '<button onclick="showNodeTab(\'compliance\')">Compliance</button>';
  document.getElementById('d_tabs').innerHTML = tabs;

  showNodeTab('overview');
  detail.classList.add('open');
}

function showNodeTab(tab) {
  const node = currentDetailNode;
  if (!node) return;

  // Update active tab button
  document.querySelectorAll('#d_tabs button').forEach(b => {
    b.classList.toggle('active', b.textContent.toLowerCase() === tab || (tab === 'overview' && b.textContent === 'Overview'));
  });

  const d = document.getElementById('d_body');

  if (tab === 'overview') {
    d.innerHTML = renderOverview(node);
  } else if (tab === 'compliance') {
    d.innerHTML = renderCompliance(node);
  } else {
    d.innerHTML = renderPersonaDetail(node, tab);
  }
}

function renderOverview(node) {
  let h = sec('Info',
    drow('Score', node.avg_score.toFixed(2), 'green') +
    drow('Generation', node.generation) +
    drow('Conversations', node.num_convos) +
    drow('Parent', node.parent_id || 'none (seed)') +
    drow('Status', node.discarded ? 'DISCARDED' : node.promoted ? 'PROMOTED' : !node.parent_id ? 'SEED' : 'Active')
  );

  // Per-persona summary
  if (node.persona_scores && Object.keys(node.persona_scores).length) {
    let ph = '';
    Object.entries(node.persona_scores).sort((a,b)=>b[1]-a[1]).forEach(([name, score]) => {
      const col = scoreColor(score);
      ph += `<div class="score-grid">
        <span class="sg-label">${name}</span>
        <div class="sg-bar"><div class="sg-fill" style="width:${score*10}%;background:${col}"></div></div>
        <span class="sg-val" style="color:${col}">${score.toFixed(1)}</span>
      </div>`;
    });
    h += sec('Per-Persona Avg', ph);
  }

  // Mutation
  if (node.components_modified && node.components_modified.length) {
    h += sec('Mutation',
      `<div style="font-size:0.85em;color:#8b949e;margin-bottom:4px">Modified: ${node.components_modified.map(c=>'<span class="tag pass">'+c+'</span>').join(' ')}</div>` +
      `<div class="mutation-text">${node.mutation_desc || 'N/A'}</div>`
    );
  }
  return h;
}

function renderPersonaDetail(node, persona) {
  const convos = (node.convo_details || []).filter(c => c.persona === persona);
  if (!convos.length) return '<div class="dim">No data for this persona</div>';

  let h = '';
  const avg = arr => arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0;
  const avgTotal = convos.map(c => c.total);
  h += `<div style="font-size:0.8em;color:#8b949e;margin-bottom:8px">${convos.length} conversations, avg: <span class="green" style="font-weight:bold">${avg(avgTotal).toFixed(2)}</span></div>`;

  const stageNames = {agent1: 'Agent 1 — Assessment (Chat)', agent2: 'Agent 2 — Resolution (Voice)', agent3: 'Agent 3 — Final Notice (Chat)'};
  const agents = ['agent1', 'agent2', 'agent3'];

  agents.forEach(a => {
    h += `<div class="stage-header">${stageNames[a]}</div>`;

    // Aggregate all checks across conversations for this agent + persona
    const goalAgg = {}, qualityAgg = {}, complianceAgg = {};

    convos.forEach(c => {
      if (!c.agents[a]) return;
      for (const [k, v] of Object.entries(c.agents[a].goal || {})) {
        if (!goalAgg[k]) goalAgg[k] = {pass:0, total:0};
        goalAgg[k].total++;
        if (v) goalAgg[k].pass++;
      }
      for (const [k, v] of Object.entries(c.agents[a].quality || {})) {
        if (!qualityAgg[k]) qualityAgg[k] = {pass:0, total:0};
        qualityAgg[k].total++;
        if (v) qualityAgg[k].pass++;
      }
      for (const [k, v] of Object.entries(c.agents[a].compliance || {})) {
        if (!complianceAgg[k]) complianceAgg[k] = {pass:0, total:0};
        complianceAgg[k].total++;
        if (v) complianceAgg[k].pass++;
      }
    });

    // Goal checks
    if (Object.keys(goalAgg).length) {
      h += `<div style="color:#d29922;font-size:0.8em;font-weight:bold;margin:6px 0 3px">Goal Completion</div>`;
      h += renderCheckAgg(goalAgg);
    }

    // Quality checks
    if (Object.keys(qualityAgg).length) {
      h += `<div style="color:#d29922;font-size:0.8em;font-weight:bold;margin:6px 0 3px">Quality</div>`;
      h += renderCheckAgg(qualityAgg);
    }

    // Compliance checks
    if (Object.keys(complianceAgg).length) {
      h += `<div style="color:#d29922;font-size:0.8em;font-weight:bold;margin:6px 0 3px">Compliance</div>`;
      h += renderCheckAgg(complianceAgg, RULE_NAMES);
    }
  });

  // Handoff checks
  h += `<div class="stage-header">Handoffs</div>`;
  ['handoff_1', 'handoff_2'].forEach(hk => {
    const label = hk === 'handoff_1' ? 'Handoff 1 (A1→A2)' : 'Handoff 2 (A1+A2→A3)';
    const agg = {};
    convos.forEach(c => {
      const hd = (c.handoffs || {})[hk] || {};
      for (const [k, v] of Object.entries(hd)) {
        if (!agg[k]) agg[k] = {pass:0, total:0};
        agg[k].total++;
        if (v) agg[k].pass++;
      }
    });
    if (Object.keys(agg).length) {
      h += `<div style="color:#58a6ff;font-size:0.85em;font-weight:bold;margin:6px 0 3px">${label}</div>`;
      h += renderCheckAgg(agg);
    }
  });

  // System checks
  h += `<div class="stage-header">System Continuity</div>`;
  const sysAgg = {};
  convos.forEach(c => {
    for (const [k, v] of Object.entries(c.system || {})) {
      if (!sysAgg[k]) sysAgg[k] = {pass:0, total:0};
      sysAgg[k].total++;
      if (v) sysAgg[k].pass++;
    }
  });
  h += renderCheckAgg(sysAgg);

  // Per-conversation totals with transcript link
  h += `<div class="stage-header">Individual Conversations</div>`;
  convos.forEach((c, i) => {
    const col = scoreColor(c.total);
    h += `<div style="display:flex;justify-content:space-between;align-items:center;padding:3px 0;font-size:0.82em">`;
    h += `<span class="dim">conv ${i+1}</span>`;
    h += `<span style="color:${col};font-weight:bold">${c.total.toFixed(2)}</span>`;
    h += `<button onclick="viewTranscript('${c.id}')" style="background:#21262d;border:1px solid #30363d;color:#58a6ff;padding:2px 8px;border-radius:3px;cursor:pointer;font-family:monospace;font-size:0.9em">View</button>`;
    h += `</div>`;
  });

  return h;
}

function renderCheckAgg(agg, nameMap) {
  let h = '';
  for (const [key, r] of Object.entries(agg)) {
    const allPass = r.pass === r.total;
    const name = (nameMap && nameMap[key]) || key.replace(/_/g, ' ');
    const icon = allPass ? '\u2714' : '\u2718';
    const col = allPass ? '#3fb950' : '#f85149';
    h += `<div class="rule-row"><span style="color:${col}">${icon} ${name}</span><span style="color:${col};font-weight:bold">${r.pass}/${r.total}</span></div>`;
  }
  return h;
}

function renderCompliance(node) {
  const comp = node.compliance_agg;
  if (!comp || !Object.keys(comp).length) return '<div class="dim">No compliance data</div>';

  let h = '';
  const agents = Object.keys(comp).sort();
  agents.forEach(agent => {
    const agentName = {agent1: 'Agent 1 — Assessment', agent2: 'Agent 2 — Resolution', agent3: 'Agent 3 — Final Notice'}[agent] || agent;
    h += `<div class="stage-header">${agentName}</div>`;
    Object.keys(comp[agent]).sort().forEach(rule => {
      const r = comp[agent][rule];
      const allPass = r.pass === r.total;
      const name = RULE_NAMES[rule] || rule;
      const icon = allPass ? '\u2714' : '\u2718';
      const col = allPass ? '#3fb950' : '#f85149';
      const pct = r.total > 0 ? (r.pass/r.total*100).toFixed(0) : 0;
      h += `<div class="rule-row"><span style="color:${col}">${icon} ${name}</span><span style="color:${col};font-weight:bold">${r.pass}/${r.total} (${pct}%)</span></div>`;
    });
  });
  return h;
}

function scoreBar(label, value) {
  const col = scoreColor(value);
  return `<div class="score-grid">
    <span class="sg-label">${label}</span>
    <div class="sg-bar"><div class="sg-fill" style="width:${value*10}%;background:${col}"></div></div>
    <span class="sg-val" style="color:${col}">${value.toFixed(1)}</span>
  </div>`;
}

function scoreColor(v) {
  return v>=8?'#3fb950':v>=6?'#58a6ff':v>=4?'#d29922':'#f85149';
}

async function showEdgeDetail(edge) {
  tooltip.style.display = 'none';
  currentDetailNode = null;

  const c = edge.child;
  const p = edge.parent;
  const diff = c.avg_score - (p ? p.avg_score : 0);

  document.getElementById('d_title').textContent = edge.from + ' \u2192 ' + edge.to;
  document.getElementById('d_tabs').innerHTML = '<button class="active">Mutation Diff</button>';

  let h = '';
  const dc = diff>0?'green':diff<0?'red':'';
  h += sec('Score Change',
    drow('Parent', (p?p.avg_score.toFixed(2):'?')) +
    drow('Child', c.avg_score.toFixed(2)) +
    drow('Difference', (diff>=0?'+':'')+diff.toFixed(2), dc) +
    drow('Conversations', c.num_convos) +
    drow('Status', c.discarded ? 'DISCARDED' : c.promoted ? 'PROMOTED' : 'Active')
  );

  h += sec('Components Modified',
    c.components_modified.map(comp => '<span class="tag pass">'+comp+'</span>').join(' ') || '<span class="dim">none</span>'
  );

  h += sec('Rationale',
    `<div class="mutation-text">${c.mutation_desc || 'N/A'}</div>`
  );

  // Fetch diff
  h += '<div class="diff-section" id="diff-content"><div class="dim">Loading diff...</div></div>';
  document.getElementById('d_body').innerHTML = h;
  detail.classList.add('open');

  const cacheKey = edge.from + '|' + edge.to;
  if (!diffCache[cacheKey]) {
    try {
      const r = await fetch(`/api/diff?parent=${encodeURIComponent(edge.from)}&child=${encodeURIComponent(edge.to)}`);
      diffCache[cacheKey] = await r.json();
    } catch(e) {
      diffCache[cacheKey] = {diffs: {}};
    }
  }

  const dd = diffCache[cacheKey];
  let dh = '';
  if (dd.diffs && Object.keys(dd.diffs).length) {
    for (const [comp, info] of Object.entries(dd.diffs)) {
      const name = comp.replace('_prompt', '');
      dh += `<div class="diff-comp">${name} (${info.old_len} \u2192 ${info.new_len} chars)</div>`;
      dh += '<div class="diff-block">';
      info.diff.split('\n').forEach(line => {
        const esc = line.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        if (line.startsWith('+++') || line.startsWith('---') || line.startsWith('@@')) {
          dh += `<div class="dl-hdr">${esc}</div>`;
        } else if (line.startsWith('+')) {
          dh += `<div class="dl-add">${esc}</div>`;
        } else if (line.startsWith('-')) {
          dh += `<div class="dl-del">${esc}</div>`;
        } else {
          dh += `<div class="dl-ctx">${esc}</div>`;
        }
      });
      dh += '</div>';
    }
  } else {
    dh = '<div class="dim">No prompt changes detected</div>';
  }
  document.getElementById('diff-content').innerHTML = dh;
}

async function viewTranscript(convId) {
  const batchParam = currentBatch ? '&batch='+currentBatch : '';
  const r = await fetch('/api/transcript?id='+encodeURIComponent(convId)+batchParam);
  const t = await r.json();
  if (t.error) {
    document.getElementById('d_body').innerHTML = `<div class="dim">${t.error}</div>`;
    return;
  }

  document.getElementById('d_title').textContent = 'Transcript: ' + convId;
  document.getElementById('d_tabs').innerHTML =
    '<button class="active" onclick="viewTranscript(\''+convId+'\')">All Stages</button>' +
    '<button onclick="viewTranscriptStage(\''+convId+'\',1)">Agent 1</button>' +
    '<button onclick="viewTranscriptStage(\''+convId+'\',2)">Agent 2</button>' +
    '<button onclick="viewTranscriptStage(\''+convId+'\',3)">Agent 3</button>';

  let h = '';
  h += `<div style="font-size:0.8em;color:#8b949e;margin-bottom:8px">Persona: <span class="blue">${t.persona_type}</span> | Outcome: <span class="${t.outcome==='deal_agreed'?'green':'yellow'}">${t.outcome}</span></div>`;

  // Show all 3 stages
  [['Agent 1 — Assessment', t.agent1, t.handoff_1],
   ['Agent 2 — Resolution', t.agent2, t.handoff_2],
   ['Agent 3 — Final Notice', t.agent3, null]].forEach(([title, msgs, handoff]) => {
    if (!msgs || !msgs.length) return;
    h += `<div class="stage-header">${title}</div>`;
    msgs.forEach(m => {
      const isAgent = m.role === 'assistant';
      const bg = isAgent ? '#0d1a28' : '#1a1a0d';
      const label = isAgent ? 'AGENT' : 'BORROWER';
      const col = isAgent ? '#58a6ff' : '#d29922';
      h += `<div style="background:${bg};padding:6px 8px;margin:3px 0;border-radius:4px;font-size:0.82em;border-left:3px solid ${col}">`;
      h += `<span style="color:${col};font-weight:bold;font-size:0.85em">${label}</span><br>`;
      h += `<span style="color:#c9d1d9">${m.content.replace(/</g,'&lt;')}</span>`;
      h += `</div>`;
    });
    if (handoff) {
      h += `<div style="background:#1a0d1a;padding:6px 8px;margin:6px 0;border-radius:4px;font-size:0.8em;border-left:3px solid #8b5cf6">`;
      h += `<span style="color:#8b5cf6;font-weight:bold">HANDOFF (${handoff.token_count} tokens)</span><br>`;
      h += `<span style="color:#a78bfa">${handoff.text.replace(/</g,'&lt;')}</span>`;
      h += `</div>`;
    }
  });

  document.getElementById('d_body').innerHTML = h;
  detail.classList.add('open');
  // Cache transcript data for stage views
  window._lastTranscript = t;
}

function viewTranscriptStage(convId, stage) {
  const t = window._lastTranscript;
  if (!t) { viewTranscript(convId); return; }

  const stageData = [null, t.agent1, t.agent2, t.agent3][stage];
  const titles = [null, 'Agent 1 — Assessment', 'Agent 2 — Resolution', 'Agent 3 — Final Notice'];
  const handoffs = [null, t.handoff_1, t.handoff_2, null];

  document.querySelectorAll('#d_tabs button').forEach((b, i) => b.classList.toggle('active', i === stage));

  let h = '';
  h += `<div class="stage-header">${titles[stage]}</div>`;
  if (stageData && stageData.length) {
    stageData.forEach(m => {
      const isAgent = m.role === 'assistant';
      const bg = isAgent ? '#0d1a28' : '#1a1a0d';
      const label = isAgent ? 'AGENT' : 'BORROWER';
      const col = isAgent ? '#58a6ff' : '#d29922';
      h += `<div style="background:${bg};padding:6px 8px;margin:3px 0;border-radius:4px;font-size:0.82em;border-left:3px solid ${col}">`;
      h += `<span style="color:${col};font-weight:bold;font-size:0.85em">${label}</span><br>`;
      h += `<span style="color:#c9d1d9">${m.content.replace(/</g,'&lt;')}</span>`;
      h += `</div>`;
    });
  } else {
    h += '<div class="dim">Agent did not run in this conversation</div>';
  }
  if (handoffs[stage]) {
    const ho = handoffs[stage];
    h += `<div style="background:#1a0d1a;padding:6px 8px;margin:6px 0;border-radius:4px;font-size:0.8em;border-left:3px solid #8b5cf6">`;
    h += `<span style="color:#8b5cf6;font-weight:bold">HANDOFF (${ho.token_count} tokens)</span><br>`;
    h += `<span style="color:#a78bfa">${ho.text.replace(/</g,'&lt;')}</span>`;
    h += `</div>`;
  }
  document.getElementById('d_body').innerHTML = h;
}

function closeDetail() {
  detail.classList.remove('open');
  selectedNode = null;
  draw();
}

function sec(title, content) {
  return `<div class="section"><h3>${title}</h3>${content}</div>`;
}
function drow(k, v, cls) {
  return `<div class="drow"><span class="dk">${k}</span><span class="dv ${cls||''}">${v}</span></div>`;
}
function trow(k, v, cls) {
  return `<div class="trow"><span class="tk">${k}</span><span class="tv ${cls||''}">${v}</span></div>`;
}

function zoomIn() { zoom = Math.min(3, zoom*1.2); draw(); }
function zoomOut() { zoom = Math.max(0.2, zoom*0.8); draw(); }
function resetView() { zoom=1; panX=0; panY=0; draw(); }

window.addEventListener('resize', draw);
load();
setInterval(load, 5000);
</script>
</body></html>"""


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == '/api/state':
            batch = qs.get('batch', [None])[0]
            self._json(get_state(batch_id=batch))
        elif parsed.path == '/api/diff':
            parent_id = qs.get('parent', [''])[0]
            child_id = qs.get('child', [''])[0]
            batch = qs.get('batch', [None])[0]
            self._json(get_diff(parent_id, child_id, batch_id=batch))
        elif parsed.path == '/api/batches':
            self._json(get_batches())
        elif parsed.path == '/api/transcript':
            conv_id = qs.get('id', [''])[0]
            batch = qs.get('batch', [None])[0]
            self._json(get_transcript(conv_id, batch_id=batch))
        elif parsed.path in ('/', '/index.html'):
            self._html(HTML)
        else:
            self.send_error(404)

    def _json(self, data):
        body = json.dumps(data, default=str).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a): pass


def get_state(batch_id: str | None = None):
    batch_dir = _resolve_batch_dir(batch_id)
    archive_file = batch_dir / "archive.json"
    costs_file = LOGS_DIR / "costs.json"  # costs are global

    variants = []
    if archive_file.exists():
        with open(archive_file) as f:
            data = json.load(f)
        for vid, e in sorted(data.items(), key=lambda x: (x[1].get('generation',0), x[0])):
            scores = e.get('scores', [])
            avg = sum(s['weighted_total'] for s in scores) / len(scores) if scores else 0
            ps = {}
            for s in scores:
                pt = s.get('persona_type', 'unknown')
                ps.setdefault(pt, []).append(s['weighted_total'])

            # Aggregate compliance: per agent, per rule → pass_count / total
            comp_agg = {}  # {agent: {rule: {pass: N, total: N}}}
            for s in scores:
                for ak, asc in s.get('agent_scores', {}).items():
                    if ak not in comp_agg:
                        comp_agg[ak] = {}
                    for rule, passed in asc.get('compliance', {}).get('rule_results', {}).items():
                        if rule not in comp_agg[ak]:
                            comp_agg[ak][rule] = {'pass': 0, 'total': 0}
                        comp_agg[ak][rule]['total'] += 1
                        if passed:
                            comp_agg[ak][rule]['pass'] += 1

            # Full per-persona, per-conversation scores for detail view
            convo_details = []
            for s in scores:
                agents_detail = {}
                for ak, asc in s.get('agent_scores', {}).items():
                    # Handle both old format (goal=float) and new format (goal={checks:{}})
                    goal_data = asc.get('goal', {})
                    quality_data = asc.get('quality', {})
                    compliance_data = asc.get('compliance', {})

                    if isinstance(goal_data, dict) and 'checks' in goal_data:
                        goal_checks = goal_data['checks']
                    else:
                        goal_checks = {'overall': goal_data >= 7 if isinstance(goal_data, (int, float)) else True}

                    if isinstance(quality_data, dict) and 'checks' in quality_data:
                        quality_checks = quality_data['checks']
                    else:
                        quality_checks = {'overall': quality_data >= 7 if isinstance(quality_data, (int, float)) else True}

                    if isinstance(compliance_data, dict) and 'rule_results' in compliance_data:
                        compliance_checks = compliance_data['rule_results']
                    else:
                        compliance_checks = {}

                    agents_detail[ak] = {
                        'goal': goal_checks,
                        'quality': quality_checks,
                        'compliance': compliance_checks,
                    }

                # Handle handoff scores — new format is dict with 'checks'
                handoff_detail = {}
                hs = s.get('handoff_scores', {})
                for hk, hv in hs.items():
                    if isinstance(hv, dict) and 'checks' in hv:
                        handoff_detail[hk] = hv['checks']
                    else:
                        handoff_detail[hk] = {'overall': hv >= 7 if isinstance(hv, (int, float)) else True}

                # System checks
                sys_data = s.get('system_checks', s.get('system_score', {}))
                if isinstance(sys_data, dict) and 'checks' in sys_data:
                    system_detail = sys_data['checks']
                else:
                    system_detail = {'overall': sys_data >= 7 if isinstance(sys_data, (int, float)) else True}

                convo_details.append({
                    'id': s.get('conversation_id', ''),
                    'persona': s.get('persona_type', 'unknown'),
                    'total': s['weighted_total'],
                    'agents': agents_detail,
                    'handoffs': handoff_detail,
                    'system': system_detail,
                })

            variants.append({
                'version_id': vid,
                'generation': e.get('generation', 0),
                'avg_score': avg,
                'num_convos': len(scores),
                'parent_id': e.get('parent_id'),
                'promoted': e.get('promoted', False),
                'discarded': e.get('discarded', False),
                'components_modified': e.get('components_modified', []),
                'mutation_desc': e.get('mutation_description', ''),
                'persona_scores': {k: sum(v)/len(v) for k, v in ps.items()},
                'compliance_agg': comp_agg,
                'convo_details': convo_details,
            })

    active = [v for v in variants if not v['discarded']]
    best = max(active, key=lambda v: v['avg_score']) if active else None
    seed = next((v for v in variants if not v['parent_id']), None)

    budget = {'spent':0,'remaining':20.0,'limit':20.0,'total_calls':0,'by_category':{}}
    if costs_file.exists():
        with open(costs_file) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                en = json.loads(line)
                budget['spent'] += en['cost_usd']
                budget['total_calls'] += 1
                c = en.get('category','?')
                budget['by_category'][c] = budget['by_category'].get(c,0) + en['cost_usd']
        budget['remaining'] = budget['limit'] - budget['spent']

    return {
        'variants': variants,
        'summary': {
            'total_variants': len(variants),
            'active': len(active),
            'discarded': len(variants)-len(active),
            'promoted': sum(1 for v in variants if v['promoted']),
            'max_generation': max((v['generation'] for v in variants), default=0),
            'best_score': best['avg_score'] if best else 0,
            'best_id': best['version_id'] if best else '',
            'seed_score': seed['avg_score'] if seed else 0,
        },
        'budget': budget,
        'best_persona_scores': best.get('persona_scores',{}) if best else {},
    }


def _resolve_batch_dir(batch_id: str | None = None) -> Path:
    """Get the batch directory — current run or specific batch."""
    if batch_id:
        d = LOGS_DIR / "runs" / batch_id
        if d.exists():
            return d
    # Try current_run symlink
    current = LOGS_DIR / "current_run"
    if current.exists():
        return current.resolve()
    # Fallback: latest run
    runs_dir = LOGS_DIR / "runs"
    if runs_dir.exists():
        dirs = sorted(runs_dir.iterdir(), reverse=True)
        if dirs:
            return dirs[0]
    # Legacy fallback
    return LOGS_DIR


def get_batches() -> list[dict]:
    """List all evolution runs."""
    runs_dir = LOGS_DIR / "runs"
    if not runs_dir.exists():
        return []
    batches = []
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        meta = {"batch_id": d.name}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        archive_path = d / "archive.json"
        if archive_path.exists():
            with open(archive_path) as f:
                data = json.load(f)
            meta["variants"] = len(data)
            if data:
                scores = []
                for e in data.values():
                    for s in e.get("scores", []):
                        scores.append(s["weighted_total"])
                meta["best_score"] = max(scores) if scores else 0
        else:
            meta["variants"] = 0
        batches.append(meta)
    return batches


def get_transcript(conv_id: str, batch_id: str | None = None) -> dict:
    """Load a conversation transcript."""
    batch_dir = _resolve_batch_dir(batch_id)
    path = batch_dir / "transcripts" / f"{conv_id}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"error": f"Transcript {conv_id} not found"}


def get_diff(parent_id: str, child_id: str, batch_id: str | None = None) -> dict:
    """Compute prompt diffs between parent and child."""
    batch_dir = _resolve_batch_dir(batch_id)
    archive_file = batch_dir / "archive.json"
    if not archive_file.exists():
        return {'error': 'No archive'}

    with open(archive_file) as f:
        data = json.load(f)

    if parent_id not in data or child_id not in data:
        return {'error': 'Version not found'}

    parent_ac = data[parent_id].get('variant_config', {}).get('agent_config', {})
    child_ac = data[child_id].get('variant_config', {}).get('agent_config', {})
    modified = data[child_id].get('components_modified', [])

    diffs = {}
    for comp in ['agent1_prompt', 'agent2_prompt', 'agent3_prompt', 'summarizer_prompt']:
        old_text = parent_ac.get(comp, '')
        new_text = child_ac.get(comp, '')
        if old_text == new_text:
            continue

        # Compute unified diff
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        diff_lines = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f'{parent_id}/{comp}',
            tofile=f'{child_id}/{comp}',
            lineterm='',
        ))

        if diff_lines:
            diffs[comp] = {
                'diff': '\n'.join(diff_lines),
                'old_len': len(old_text),
                'new_len': len(new_text),
                'was_modified': comp in modified,
            }

    return {
        'parent': parent_id,
        'child': child_id,
        'components_modified': modified,
        'mutation_desc': data[child_id].get('mutation_description', ''),
        'diffs': diffs,
    }


if __name__ == '__main__':
    port = 8050
    print(f"Dashboard: http://localhost:{port}")
    HTTPServer(('0.0.0.0', port), Handler).serve_forever()
