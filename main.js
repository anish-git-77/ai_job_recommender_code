/* ── Tab switching ──────────────────────────── */
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

  if (tab === 'upload') {
    document.querySelectorAll('.tab')[0].classList.add('active');
    document.getElementById('tab-upload').classList.add('active');
  } else {
    document.querySelectorAll('.tab')[1].classList.add('active');
    document.getElementById('tab-text').classList.add('active');
  }
}

/* ── Drag & drop ────────────────────────────── */
let selectedFile = null;

function handleDragOver(e) {
  e.preventDefault();
  document.getElementById('dropZone').classList.add('dragover');
}

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('dropZone').classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
}

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) setFile(file);
}

function setFile(file) {
  const allowed = ['application/pdf', 'text/plain'];
  const ext = file.name.split('.').pop().toLowerCase();
  if (!['pdf', 'txt'].includes(ext)) {
    showToast('Only PDF and TXT files are supported.');
    return;
  }
  selectedFile = file;
  const lbl = document.getElementById('fileLabel');
  lbl.textContent = `✓ Selected: ${file.name} (${(file.size/1024).toFixed(1)} KB)`;
  lbl.classList.remove('hidden');
}

/* ── Submit upload ──────────────────────────── */
async function submitUpload() {
  if (!selectedFile) { showToast('Please select a file first.'); return; }

  const topK = document.getElementById('topK-upload').value;
  const form = new FormData();
  form.append('resume', selectedFile);
  form.append('top_k', topK);

  showLoading();
  try {
    const res  = await fetch('/upload', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) { showToast(data.error || 'Upload failed'); return; }
    renderResults(data);
  } catch (err) {
    showToast('Network error. Is the Flask server running?');
  } finally {
    hideLoading();
  }
}

/* ── Submit text ────────────────────────────── */
async function submitText() {
  const text = document.getElementById('skillsInput').value.trim();
  if (!text) { showToast('Please enter some text.'); return; }

  const topK = document.getElementById('topK-text').value;

  showLoading();
  try {
    const res  = await fetch('/recommend-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, top_k: Number(topK) }),
    });
    const data = await res.json();
    if (!res.ok) { showToast(data.error || 'Request failed'); return; }
    renderResults(data);
  } catch (err) {
    showToast('Network error. Is the Flask server running?');
  } finally {
    hideLoading();
  }
}

/* ── Render results ─────────────────────────── */
function renderResults(data) {
  const resultsSection = document.getElementById('results');
  resultsSection.classList.remove('hidden');

  // Profile
  const profile = data.profile;
  const profileHTML = `
    <div class="profile-stat">
      <div class="stat-label">Detected Skills</div>
      <div class="stat-value">${profile.detected_skills.length} found</div>
      <div class="skills-wrap">
        ${profile.detected_skills.slice(0, 12).map(s =>
          `<span class="skill-chip">${s}</span>`).join('')}
      </div>
    </div>
    <div class="profile-stat">
      <div class="stat-label">Experience</div>
      <div class="stat-value">${profile.experience_years > 0 ? profile.experience_years + ' years' : 'Not detected'}</div>
    </div>
    <div class="profile-stat">
      <div class="stat-label">Resume Length</div>
      <div class="stat-value">${profile.word_count} words</div>
    </div>
  `;
  document.getElementById('profileContent').innerHTML = profileHTML;

  // Job cards
  const jobs = data.jobs;
  if (!jobs || jobs.length === 0) {
    document.getElementById('jobCards').innerHTML = '<p style="color:var(--muted)">No jobs found. Try adding more detail to your resume.</p>';
    return;
  }

  const cardsHTML = jobs.map(job => {
    const scoreClass = job.match_score >= 70 ? 'score-high' : job.match_score >= 50 ? 'score-med' : 'score-low';
    const matchedSkills = job.matched_skills || [];
    const allSkills = job.skills.split(',').map(s => s.trim());

    const skillsHTML = allSkills.map(sk => {
      const isMatch = matchedSkills.some(m => m.toLowerCase() === sk.toLowerCase());
      return `<span class="skill-chip ${isMatch ? 'matched' : ''}">${sk}</span>`;
    }).join('');

    return `
      <div class="job-card">
        <div class="job-rank"><span class="rank-badge">#${job.rank}</span></div>
        <div class="job-score ${scoreClass}">${job.match_score}%</div>
        <div class="score-label">Match Score</div>

        <div class="job-title">${job.title}</div>
        <div class="job-company">🏢 ${job.company}</div>
        <div class="job-meta">
          <span class="meta-tag">📍 ${job.location}</span>
          <span class="meta-tag">🎯 ${job.experience_level}</span>
          <span class="meta-tag">💰 ${job.salary_range}</span>
          <span class="meta-tag">🛠 ${job.skill_match_pct}% skills match</span>
        </div>
        <div class="job-desc">${job.description.substring(0, 140)}…</div>
        <div class="job-skills">
          <div class="skills-heading">Skills (🟢 = matched):</div>
          <div class="skills-wrap">${skillsHTML}</div>
        </div>
      </div>
    `;
  }).join('');

  document.getElementById('jobCards').innerHTML = cardsHTML;

  // Scroll to results
  resultsSection.scrollIntoView({ behavior: 'smooth' });
}

/* ── Jobs browser ───────────────────────────── */
async function loadAllJobs() {
  try {
    const res  = await fetch('/jobs');
    const jobs = await res.json();

    const cols = ['job_id','title','company','location','experience_level','salary_range','skills'];
    const headers = cols.map(c => `<th>${c.replace('_',' ')}</th>`).join('');
    const rows = jobs.map(j =>
      `<tr>${cols.map(c => `<td>${j[c] ?? ''}</td>`).join('')}</tr>`
    ).join('');

    document.getElementById('allJobsTable').innerHTML =
      `<table><thead><tr>${headers}</tr></thead><tbody>${rows}</tbody></table>`;
  } catch (e) {
    showToast('Could not load jobs.');
  }
}

/* ── Helpers ────────────────────────────────── */
function showLoading() {
  document.getElementById('loading').classList.remove('hidden');
  document.getElementById('results').classList.add('hidden');
  window.scrollTo({ top: document.getElementById('loading').offsetTop - 80, behavior: 'smooth' });
}
function hideLoading() {
  document.getElementById('loading').classList.add('hidden');
}
function showToast(msg) {
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}
