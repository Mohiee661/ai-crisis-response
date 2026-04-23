const API = 'http://localhost:8000';
let logCount = 0;
let lastDecision = null;
let lastLifecycle = null;

// ── Manual Override ──
async function override(action) {
  try {
    await fetch(`${API}/override`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action })
    });
  } catch (e) { console.error('Override failed', e); }
}

// ── Main Sync Loop ──
async function sync() {
  try {
    const res = await fetch(`${API}/status`);
    const d = await res.json();
    render(d);
  } catch (e) { }
}

function render(d) {
  // Video & Location
  if (d.frame) document.getElementById('live-feed').src = `data:image/jpeg;base64,${d.frame}`;
  document.getElementById('cam-tag').textContent = `CAM_1 | ${d.location || 'Unknown'}`;
  document.getElementById('playlist-tag').textContent = `🎥 Now Playing: ${d.current_video || 'None'}`;

  // Lifecycle & Status
  const lb = document.getElementById('lifecycle-badge');
  if (d.lifecycle_state !== lastLifecycle) {
    lastLifecycle = d.lifecycle_state;
    lb.className = `lifecycle-badge ${d.lifecycle_state}`;
    lb.textContent = d.lifecycle_state;
  }

  const pill = document.getElementById('status-pill');
  pill.className = `status-pill ${d.status}`;
  document.getElementById('status-label').textContent = d.status === 'ALERT' ? 'THREAT DETECTED' : 'System Secure';

  // Decision Intelligence
  if (d.decision && d.decision !== lastDecision) {
    lastDecision = d.decision;
    const isFall = d.decision.includes('AMBULANCE');
    const title = isFall ? '🚑 MEDICAL EMERGENCY' : '🚒 FIRE EMERGENCY';
    
    document.getElementById('decision-slot').innerHTML = `
      <div class="decision-card ${d.severity || 'HIGH'}">
        <div class="decision-title">${title}</div>
        <div class="decision-loc">${d.location}</div>
        <div style="font-size: 0.75rem; margin-bottom: 8px;">Outcome: <b>${d.decision}</b></div>
        <div class="section-label">Confidence Explanation</div>
        <div class="reasoning-list">
          ${(d.confidence_explanation || []).map(ex => `<div class="reasoning-item">${ex}</div>`).join('')}
        </div>
        <div style="margin-top: 16px; font-size: 0.6rem; color: var(--text-muted); font-family: var(--font-mono);">
          Decision Reason: ${d.decision_reason || 'N/A'}
        </div>
      </div>
    `;
  } else if (!d.decision) {
    document.getElementById('decision-slot').innerHTML = '<div class="safe-state">NO ACTIVE THREATS</div>';
  }

  // LLM Social Summary
  const socialSlot = document.getElementById('social-summary-slot');
  if (d.llm_summary) {
    socialSlot.innerHTML = `
      <div class="social-summary">"${d.llm_summary}"</div>
      <a href="${d.llm_link}" target="_blank" class="social-link">View Source Intelligence →</a>
      <div style="margin-top: 8px; font-size: 0.6rem; color: var(--success); font-weight: 800;">
        Confirmation: ${d.llm_confirmation}
      </div>
    `;
    socialSlot.style.opacity = '1';
  } else {
    socialSlot.innerHTML = '<div class="safe-state">Awaiting automated confirmation...</div>';
    socialSlot.style.opacity = '0.4';
  }

  // System Health
  if (d.system_health) {
    document.getElementById('health-models').textContent = d.system_health.model_status;
    document.getElementById('health-camera').textContent = d.system_health.camera_status;
    document.getElementById('health-api').textContent = d.system_health.api_status;
    document.getElementById('health-latency').textContent = d.system_health.latency;
  }

  // Logs
  const logs = d.logs || [];
  if (logs.length !== logCount) {
    const box = document.getElementById('logs-box');
    logs.slice(logCount).forEach(l => {
      const el = document.createElement('div');
      el.className = 'log-line';
      const tag = l.match(/\[(.*?)\]/)?.[1] || 'SYS';
      el.innerHTML = `<span class="log-tag">${tag}</span>${l.replace(`[${tag}] `, '')}`;
      box.prepend(el);
    });
    logCount = logs.length;
  }
}

setInterval(sync, 400);
sync();
