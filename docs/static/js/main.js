// ── CONFIG ────────────────────────────────────────────────────────────────
const APPS_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbzDfs7_90-ses_2cNxUfrOFzucSTNZd6DrSMSgnQdfetqnMxcnSyL0y1WHs0Kcgc-m4/exec"; // ← paste your URL here

const STUDY_ID        = "appearance";
const USERNAME_KEY    = 'spaceflow_username'; // shared with landing page + tau study
const TRIALS_PER_PARTICIPANT = 8;

// ── STATE ─────────────────────────────────────────────────────────────────
let trials        = [];
let trialIndex    = 0;
let currentTrial  = null;
// Check localStorage first (set by landing page), fall back to sessionStorage
let currentUsername =
  localStorage.getItem(USERNAME_KEY) ||
  sessionStorage.getItem(`username_${STUDY_ID}`);

// ── USERNAME MODAL ────────────────────────────────────────────────────────
if (currentUsername) {
  hideModal();
  initStudy();
} else {
  document.getElementById('username-modal').style.display = 'flex';
}

document.getElementById('username-submit').addEventListener('click', submitUsername);
document.getElementById('username-input').addEventListener('keypress', e => {
  if (e.key === 'Enter') submitUsername();
});

function submitUsername() {
  const input    = document.getElementById('username-input');
  const errorDiv = document.getElementById('username-error');
  const username = input.value.trim();
  if (!username) {
    errorDiv.textContent = 'Please enter a name to continue.';
    return;
  }
  localStorage.setItem(USERNAME_KEY, username);
  sessionStorage.setItem(`username_${STUDY_ID}`, username);
  currentUsername = username;
  document.getElementById('display-username').textContent = username;
  hideModal();
  initStudy();
}

function hideModal() {
  document.getElementById('username-modal').style.display = 'none';
  document.getElementById('main-content').style.display  = 'block';
  document.getElementById('display-username').textContent = currentUsername || '';
}

// ── STUDY INIT ────────────────────────────────────────────────────────────
async function initStudy() {
  try {
    const res = await fetch('./trials.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const allTrials = await res.json();
    const shuffled  = [...allTrials].sort(() => Math.random() - 0.5);
    trials     = shuffled.slice(0, Math.min(TRIALS_PER_PARTICIPANT, shuffled.length));
    trialIndex = 0;
    loadNextTrial();
  } catch (err) {
    console.error('Failed to load trials.json:', err);
    showMessage('Could not load study trials. Please refresh the page.', true);
  }
}

// ── LOAD TRIAL ────────────────────────────────────────────────────────────
function loadNextTrial() {
  if (trialIndex >= trials.length) { showCompletion(); return; }
  currentTrial = trials[trialIndex++];
  updateProgress();
  populateTrial(currentTrial);
}

function populateTrial(trial) {
  document.getElementById('prompt-box').textContent = trial.prompt || '—';

  document.getElementById('ref-a-img').src          = trial.ref_a;
  document.getElementById('ref-b-img').src          = trial.ref_b;
  document.getElementById('ref-a-label').textContent = `Reference A: ${trial.ref_a_label}`;
  document.getElementById('ref-b-label').textContent = `Reference B: ${trial.ref_b_label}`;
  document.getElementById('q1-part').textContent     = trial.ref_a_label;
  document.getElementById('q2-part').textContent     = trial.ref_b_label;

  fillOutputs('imgs-a', trial.outputs_a);
  fillOutputs('imgs-b', trial.outputs_b);

  setSpan('scene',          trial.scene_id);
  setSpan('model_a_method', trial.mapping.A);
  setSpan('model_b_method', trial.mapping.B);

  document.querySelectorAll('#survey-form input[type="radio"]')
          .forEach(r => r.checked = false);
  document.getElementById('form-error').style.display = 'none';
  document.getElementById('message').style.display    = 'none';

  const btn = document.getElementById('submit-btn');
  btn.disabled    = false;
  btn.textContent = 'Submit & Next →';
}

// ── MEDIA HELPERS ─────────────────────────────────────────────────────────
function fillOutputs(containerId, urls) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  const altText = containerId === 'imgs-a' ? 'Sample A' : 'Sample B';
  (urls || []).forEach(url => container.appendChild(createMediaElement(url, altText)));
}

function createMediaElement(url, altText) {
  const ext = url.split('.').pop().toLowerCase();

  if (ext === 'glb' || ext === 'gltf') {
    const mv = document.createElement('model-viewer');
    mv.setAttribute('src', url);
    mv.setAttribute('auto-rotate', '');
    mv.setAttribute('camera-controls', '');
    mv.setAttribute('alt', altText);
    mv.setAttribute('shadow-intensity', '1');
    mv.setAttribute('orientation', '0deg -90deg 0deg');
    mv.style.width  = '100%';
    mv.style.height = '350px';
    return mv;
  } else if (ext === 'mp4' || ext === 'webm') {
    const video = document.createElement('video');
    video.src         = url;
    video.autoplay    = true;
    video.loop        = true;
    video.muted       = true;
    video.playsInline = true;
    video.style.cssText = 'width:100%;border-radius:6px;display:block;';
    return video;

  } else {
    const img     = document.createElement('img');
    img.src       = url;
    img.alt       = altText;
    img.className = 'model-image';
    img.loading   = 'lazy';
    return img;
  }
}

function setSpan(id, value) {
  const el       = document.getElementById(id);
  el.textContent = value;
  el.dataset.value = value;
}

// ── PROGRESS ──────────────────────────────────────────────────────────────
function updateProgress() {
  const pct = ((trialIndex - 1) / trials.length) * 100;
  document.getElementById('progress-text').textContent =
    `Trial ${trialIndex} of ${trials.length}`;
  document.getElementById('progress-fill').style.width = `${pct}%`;
}

// ── SUBMIT ────────────────────────────────────────────────────────────────
document.getElementById('survey-form').addEventListener('submit', async (e) => {
  e.preventDefault();

  const answers = {};
  let allAnswered = true;
  document.querySelectorAll('#survey-form .question').forEach((qDiv, idx) => {
    const name    = `q${idx + 1}`;
    const checked = qDiv.querySelector(`input[name="${name}"]:checked`);
    answers[name] = checked ? checked.value : null;
    if (!checked) allAnswered = false;
  });

  if (!allAnswered) {
    document.getElementById('form-error').style.display = 'block';
    return;
  }
  document.getElementById('form-error').style.display = 'none';

  const btn = document.getElementById('submit-btn');
  btn.disabled    = true;
  btn.textContent = 'Saving…';

  const payload = {
    study_id:     STUDY_ID,
    username:     currentUsername,
    scene_id:     document.getElementById('scene').dataset.value,
    model_a_name: document.getElementById('model_a_method').dataset.value,
    model_b_name: document.getElementById('model_b_method').dataset.value,
    answers,
    timestamp:    new Date().toISOString(),
  };

  try {
    const params = new URLSearchParams({ data: JSON.stringify(payload) });
    await fetch(`${APPS_SCRIPT_URL}?${params.toString()}`, {
      method: 'GET',
      mode:   'no-cors',
    });
    await new Promise(r => setTimeout(r, 300));
    loadNextTrial();
  } catch (err) {
    console.error('Submit error:', err);
    showMessage('Error saving — please check your connection and try again.', true);
    btn.disabled    = false;
    btn.textContent = 'Submit & Next →';
  }
});

// ── COMPLETION ────────────────────────────────────────────────────────────
function showCompletion() {
  document.getElementById('progress-fill').style.width = '100%';
  document.getElementById('progress-text').textContent =
    `${trials.length} of ${trials.length} — Complete`;

  document.querySelector('.task-card').style.display        = 'none';
  document.querySelector('.references-section').style.display = 'none';
  document.querySelector('.outputs-section').style.display  = 'none';
  document.querySelector('.survey-section').style.display   = 'none';

  document.querySelector('.page').insertAdjacentHTML('beforeend', `
    <div class="completion-card">
      <div class="completion-emoji">🎉</div>
      <div class="completion-title">Study complete!</div>
      <div class="completion-text">
        Thank you for participating, <strong>${currentUsername}</strong>.<br>
        Your responses have been saved.<br><br>
        <a href="./index.html" style="color:var(--accent);font-weight:500;">
          ← Back to study selection
        </a>
      </div>
    </div>
  `);
}

// ── HELPERS ───────────────────────────────────────────────────────────────
function showMessage(msg, isError = false) {
  const box     = document.getElementById('message');
  box.className = 'message ' + (isError ? 'error' : 'success');
  box.textContent = msg;
  box.style.display = 'block';
}