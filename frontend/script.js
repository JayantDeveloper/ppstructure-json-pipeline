const fileInput = document.getElementById('file-input');
const preview = document.getElementById('preview');
const runBtn = document.getElementById('run-btn');
const output = document.getElementById('output');
const statusEl = document.getElementById('status');
const dropZone = document.getElementById('drop-zone');

let selectedFile = null;

function loadFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  preview.style.display = 'block';
  runBtn.disabled = false;
  output.textContent = 'Results will appear here.';
  setStatus('');
}

fileInput.addEventListener('change', () => loadFile(fileInput.files[0]));

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor = '#2563eb'; });
dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = '#bbb'; });
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.style.borderColor = '#bbb';
  loadFile(e.dataTransfer.files[0]);
});

runBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  runBtn.disabled = true;
  setStatus('Running OCR…');
  output.textContent = '';

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const res = await fetch('/ocr', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    output.textContent = JSON.stringify(data, null, 2);
    setStatus('Done.');
  } catch (err) {
    output.textContent = '';
    setStatus('Error: ' + err.message, true);
  } finally {
    runBtn.disabled = false;
  }
});

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.className = isError ? 'error' : '';
}
