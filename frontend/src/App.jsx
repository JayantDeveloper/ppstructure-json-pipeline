import { useEffect, useRef, useState } from 'react';
import { jsPDF } from 'jspdf';

const EMPTY_SUMMARY = {
  engine: 'PP-StructureV3',
  page_count: 0,
  table_count: 0,
  field_count: 0,
  text_characters: 0,
};

const OCR_MODES = [
  {
    value: 'fast',
    label: 'Fast',
    description: 'Safer on older Macs. Uses smaller models, downscales oversized images, and skips table recognition.',
  },
  {
    value: 'full',
    label: 'Full',
    description: 'More accurate but heavier. Uses larger models and keeps table recognition on.',
  },
];

function createEmptyResult() {
  return {
    summary: EMPTY_SUMMARY,
    structured_fields: [],
    tables: [],
    markdown: 'Run the pipeline to view markdown output.',
    raw_text: '',
    pages: [],
    processing: null,
    documents: [],
  };
}

export default function App() {
  const copyTimerRef = useRef(null);
  const runStartRef = useRef(null);
  const reportUrlRef = useRef('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [badgeStates, setBadgeStates] = useState([]);
  const [activePreviewIndex, setActivePreviewIndex] = useState(0);
  const [processingPreviewIndex, setProcessingPreviewIndex] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [previewUrls, setPreviewUrls] = useState([]);
  const [ocrMode, setOcrMode] = useState('fast');
  const [elapsedMs, setElapsedMs] = useState(0);
  const [lastRunMs, setLastRunMs] = useState(0);
  const [status, setStatus] = useState({ message: '', isError: false, showSpinner: false });
  const [result, setResult] = useState(createEmptyResult);
  const [copiedKey, setCopiedKey] = useState('');
  const [reportState, setReportState] = useState({ message: '', isError: false, isGenerating: false });

  const fileMeta = getFileMeta(selectedFiles);
  const currentMode = OCR_MODES.find((mode) => mode.value === ocrMode) || OCR_MODES[0];
  const summary = result.summary || EMPTY_SUMMARY;
  const reportDocuments = getDocumentEntries(result);
  const hasStructuredFields = result.structured_fields.length > 0;
  const displayTables = dedupeTables(result.tables);
  const canOpenReport = !isRunning && selectedFiles.length > 0 && reportDocuments.length > 0;
  const highlightedPreviewIndex = processingPreviewIndex ?? activePreviewIndex;

  useEffect(() => {
    if (!selectedFiles.length) {
      setPreviewUrls([]);
      return undefined;
    }

    const urls = selectedFiles.map((f) => URL.createObjectURL(f));
    setPreviewUrls(urls);

    return () => {
      urls.forEach((u) => URL.revokeObjectURL(u));
    };
  }, [selectedFiles]);

  useEffect(() => {
    return () => {
      if (copyTimerRef.current) {
        clearTimeout(copyTimerRef.current);
      }

      if (reportUrlRef.current) {
        URL.revokeObjectURL(reportUrlRef.current);
        reportUrlRef.current = '';
      }
    };
  }, []);

  useEffect(() => {
    if (!isRunning || !runStartRef.current) {
      return undefined;
    }

    const updateElapsed = () => {
      setElapsedMs(Date.now() - runStartRef.current);
    };

    updateElapsed();
    const intervalId = window.setInterval(updateElapsed, 1000);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [isRunning]);

  function handleFiles(rawFiles) {
    if (isRunning) {
      return;
    }

    const files = Array.from(rawFiles || []).filter(isSupportedFile);
    if (!files.length) {
      setStatus({
        message: 'Upload a PDF or image file.',
        isError: true,
        showSpinner: false,
      });
      return;
    }

    setSelectedFiles(files);
    setBadgeStates(files.map(() => 'queued'));
    setActivePreviewIndex(0);
    setProcessingPreviewIndex(null);
    setElapsedMs(0);
    setLastRunMs(0);
    setResult(createEmptyResult());
    setStatus({ message: '', isError: false, showSpinner: false });
    setReportState({ message: '', isError: false, isGenerating: false });

    if (reportUrlRef.current) {
      URL.revokeObjectURL(reportUrlRef.current);
      reportUrlRef.current = '';
    }
  }

  function handleInputChange(event) {
    handleFiles(event.target.files);
    event.target.value = '';
  }

  function handleDragOver(event) {
    event.preventDefault();
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = 'copy';
    }
    setDragging(true);
  }

  function handleDragLeave(event) {
    if (!event.currentTarget.contains(event.relatedTarget)) {
      setDragging(false);
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    setDragging(false);

    if (isRunning) {
      return;
    }

    handleFiles(getDroppedFiles(event.dataTransfer));
  }

  function handleFileKeyDown(event, index) {
    if (event.key !== 'Enter' && event.key !== ' ') {
      return;
    }

    event.preventDefault();
    setActivePreviewIndex(index);
  }

  async function handleRun() {
    if (!selectedFiles.length || isRunning) {
      return;
    }

    const filesToProcess = selectedFiles;
    const allResults = [];
    let errorCount = 0;

    runStartRef.current = Date.now();
    setIsRunning(true);
    setElapsedMs(0);
    setLastRunMs(0);
    setProcessingPreviewIndex(0);
    setBadgeStates(filesToProcess.map(() => 'queued'));
    setReportState({ message: '', isError: false, isGenerating: false });

    for (let index = 0; index < filesToProcess.length; index += 1) {
      const file = filesToProcess[index];
      setProcessingPreviewIndex(index);
      setBadgeStates((current) => current.map((state, itemIndex) => (
        itemIndex === index ? 'running' : state
      )));
      setStatus({
        message: filesToProcess.length > 1
          ? `Processing ${index + 1} of ${filesToProcess.length}: ${file.name}...`
          : 'Running PP-StructureV3. The first run may take several minutes while models download.',
        isError: false,
        showSpinner: true,
      });

      const formData = new FormData();
      formData.append('file', file);
      formData.append('ocr_mode', ocrMode);

      try {
        const response = await fetch('/ocr', { method: 'POST', body: formData });
        if (!response.ok) {
          const errorPayload = await response.json().catch(() => ({ detail: response.statusText }));
          throw new Error(errorPayload.detail || response.statusText);
        }

        const data = normalizeRunResult(await response.json());
        allResults.push(data);
        setBadgeStates((current) => current.map((state, itemIndex) => (
          itemIndex === index ? 'done' : state
        )));
      } catch (error) {
        errorCount += 1;
        setBadgeStates((current) => current.map((state, itemIndex) => (
          itemIndex === index ? 'error' : state
        )));
        setStatus({
          message: `Error on ${file.name}: ${error.message}`,
          isError: true,
          showSpinner: false,
        });
      }
    }

    const runDuration = runStartRef.current ? Date.now() - runStartRef.current : 0;
    runStartRef.current = null;
    setElapsedMs(runDuration);
    setLastRunMs(runDuration);
    setIsRunning(false);
    setProcessingPreviewIndex(null);

    if (!allResults.length) {
      return;
    }

    setResult(mergeResults(allResults));
    setStatus({
      message: errorCount > 0
        ? `Done. ${allResults.length} succeeded, ${errorCount} failed.`
        : filesToProcess.length > 1
          ? `All ${filesToProcess.length} files processed.`
          : 'Extraction complete.',
      isError: errorCount > 0,
      showSpinner: false,
    });
  }

  async function copyText(key, content) {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedKey(key);
      if (copyTimerRef.current) {
        clearTimeout(copyTimerRef.current);
      }
      copyTimerRef.current = window.setTimeout(() => {
        setCopiedKey('');
      }, 1500);
    } catch {
      // Clipboard access is optional in this demo.
    }
  }

  async function handleOpenReport() {
    if (!canOpenReport || reportState.isGenerating) {
      return;
    }

    const previewTab = window.open('', '_blank');
    if (previewTab) {
      previewTab.document.title = 'Preparing OCR report';
      previewTab.document.body.innerHTML = '<div style="font-family: Avenir Next, Segoe UI, sans-serif; padding: 32px; color: #16223b;">Preparing PDF report...</div>';
    }

    setReportState({ message: '', isError: false, isGenerating: true });

    try {
      const pdfBlob = await buildReportPdf({
        documents: reportDocuments,
        files: matchFilesToDocuments(selectedFiles, reportDocuments),
        summary,
        lastRunMs,
        modeLabel: result.processing?.ocr_mode_label || currentMode.label,
      });

      if (reportUrlRef.current) {
        URL.revokeObjectURL(reportUrlRef.current);
      }

      const reportUrl = URL.createObjectURL(pdfBlob);
      reportUrlRef.current = reportUrl;

      if (previewTab) {
        previewTab.location.replace(reportUrl);
      } else if (!window.open(reportUrl, '_blank')) {
        throw new Error('Allow pop-ups to open the PDF report.');
      }

      setReportState({
        message: 'Report PDF opened in a new tab.',
        isError: false,
        isGenerating: false,
      });
    } catch (error) {
      if (previewTab && !previewTab.closed) {
        previewTab.close();
      }

      setReportState({
        message: error?.message || 'Unable to build the PDF report.',
        isError: true,
        isGenerating: false,
      });
    }
  }

  return (
    <main className="page-shell">
      <section className="hero">
        <p className="eyebrow">OCR Demo</p>
        <h1>Pipeline 4 Structured Extraction</h1>
        <p className="lede">
          Upload a screenshot or PDF and inspect the structured output produced by
          {' '}
          <strong>PP-StructureV3</strong>
          .
        </p>
      </section>

      <section className="top-grid">
        <div className="panel">
          <div className="panel-header">
            <div>
              <p className="section-kicker">Input</p>
              <h2>Upload Document</h2>
            </div>
            <button type="button" onClick={handleRun} disabled={!selectedFiles.length || isRunning}>
              {isRunning ? 'Running...' : 'Run P4'}
            </button>
          </div>

          <div
            className={`upload-area${dragging ? ' dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              id="file-input"
              type="file"
              accept="image/*,.pdf,application/pdf"
              multiple
              disabled={isRunning}
              onChange={handleInputChange}
            />
            <label htmlFor="file-input">
              Drop images or PDFs here
              <span>or browse your files — multiple files supported</span>
            </label>
          </div>

          <div className="mode-strip">
            <div className="mode-copy">
              <p className="mode-label">OCR Mode</p>
              <p className="mode-description">{currentMode.description}</p>
            </div>
            <div className="mode-toggle" role="tablist" aria-label="OCR mode">
              {OCR_MODES.map((mode) => (
                <button
                  key={mode.value}
                  type="button"
                  className={`mode-toggle-btn${ocrMode === mode.value ? ' active' : ''}`}
                  onClick={() => setOcrMode(mode.value)}
                  disabled={isRunning}
                >
                  {mode.label}
                </button>
              ))}
            </div>
          </div>

          <div className="file-meta">{fileMeta}</div>
          {selectedFiles.length > 0 ? (
            <div className="file-queue">
              {selectedFiles.length > 1 && (
                <p className="file-queue-label">
                  Queue
                  <span>{selectedFiles.length} files</span>
                </p>
              )}
              <ul className="file-list">
                {selectedFiles.map((file, index) => {
                  const state = badgeStates[index] || 'queued';
                  return (
                    <li
                      key={`${file.name}-${file.size}-${index}`}
                      className={`file-list-item${index === highlightedPreviewIndex ? ' active' : ''}`}
                      tabIndex={0}
                      role="button"
                      aria-pressed={index === highlightedPreviewIndex}
                      onClick={() => setActivePreviewIndex(index)}
                      onKeyDown={(event) => handleFileKeyDown(event, index)}
                    >
                      <span className="file-list-num">{String(index + 1).padStart(2, '0')}</span>
                      <FileIcon file={file} />
                      <span className="file-list-name">{file.name}</span>
                      <span className="file-list-size">{formatBytes(file.size)}</span>
                      <span className="file-list-status" data-state={state}>
                        <span className="file-list-dot" />
                        <span className="file-list-status-label">{state}</span>
                      </span>
                    </li>
                  );
                })}
              </ul>
            </div>
          ) : null}

          <div className="preview-shell">
            {!selectedFiles.length ? (
              <div className="empty-preview">Preview will appear here.</div>
            ) : (
              <div className="thumb-grid">
                {selectedFiles.map((file, index) => (
                  <button
                    key={`${file.name}-${index}`}
                    type="button"
                    className={`thumb-cell${index === highlightedPreviewIndex ? ' active' : ''}`}
                    onClick={() => setActivePreviewIndex(index)}
                    title={file.name}
                  >
                    {isPdfFile(file) ? (
                      <div className="thumb-pdf-placeholder">PDF</div>
                    ) : (
                      <img
                        className="thumb-img"
                        src={previewUrls[index] || ''}
                        alt={file.name}
                      />
                    )}
                    <span className="thumb-label">{file.name}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="status-row">
            <div className={status.isError ? 'status error' : 'status'}>
              <span className={`spinner${status.showSpinner ? ' active' : ''}`} />
              <span>{status.message}</span>
            </div>
            {(isRunning || lastRunMs > 0) ? (
              <div className="timer-chip">
                <span className="timer-chip-label">{isRunning ? 'Elapsed' : 'Last run'}</span>
                <strong>{formatDuration(isRunning ? elapsedMs : lastRunMs)}</strong>
              </div>
            ) : null}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <div>
              <p className="section-kicker">Snapshot</p>
              <h2>Extraction Summary</h2>
            </div>
          </div>
          <div className="summary-grid">
            <StatCard label="Engine" value={summary.engine || 'PP-StructureV3'} />
            <StatCard label="Pages" value={summary.page_count ?? 0} />
            <StatCard label="Tables" value={summary.table_count ?? 0} />
            <StatCard label="Fields" value={summary.field_count ?? 0} />
            <StatCard label="Text Chars" value={summary.text_characters ?? 0} />
          </div>
          <p className="summary-note">
            The backend returns markdown, structured fields, detected tables, and page-level metadata.
          </p>
          <div className="summary-actions">
            <button
              type="button"
              className="summary-report-btn"
              onClick={handleOpenReport}
              disabled={!canOpenReport || reportState.isGenerating}
            >
              {reportState.isGenerating ? 'Preparing PDF...' : 'Open Report PDF'}
            </button>
            <p className={reportState.isError ? 'summary-feedback error' : 'summary-feedback'}>
              {reportState.message || 'Creates a readable PDF with each source image and its OCR output.'}
            </p>
          </div>
        </div>
      </section>

      <section className="results-grid">
        <div className="panel tables-panel">
          <div className="panel-header">
            <div>
              <p className="section-kicker">Structured Regions</p>
              <h2>Detected Tables</h2>
            </div>
          </div>
          <div className="tables-stack">
            {displayTables.length > 0 ? displayTables.map((table, index) => (
              <article key={`table-${index}`} className="mini-table-card">
                <div className="mini-table-head">
                  <strong>{`Table ${index + 1}`}</strong>
                  <span>{`Page ${(table.page_index ?? 0) + 1}`}</span>
                  <span>{`BBox ${Array.isArray(table.bbox) ? table.bbox.join(', ') : '0, 0, 0, 0'}`}</span>
                </div>
                <div className="table-shell">
                  {(table.cells || []).length > 0 ? (
                    <table className="results-table compact">
                      <thead>
                        <tr>
                          <th>Field</th>
                          <th>Value</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(table.cells || []).map((cell, cellIndex) => (
                          <tr key={`cell-${cellIndex}`}>
                            <td>{cell.field || ''}</td>
                            <td>{cell.value || ''}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (table.rows || []).length > 0 ? (
                    <table className="results-table compact generic-table">
                      <tbody>
                        {(table.rows || []).map((row, rowIndex) => (
                          <tr key={`row-${rowIndex}`}>
                            {row.map((cell, cellIndex) => (
                              <td key={`cell-${rowIndex}-${cellIndex}`}>{cell || ''}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <table className="results-table compact">
                      <tbody>
                        <tr className="empty-row">
                          <td colSpan="2">No parsed cells.</td>
                        </tr>
                      </tbody>
                    </table>
                  )}
                </div>
              </article>
            )) : (
              <p className="empty-copy">
                {result.processing?.ocr_mode === 'fast'
                  ? 'Fast mode skips table recognition to reduce load.'
                  : 'Detected tables will appear here.'}
              </p>
            )}
          </div>
        </div>

        {hasStructuredFields ? (
          <div className="panel fields-panel">
            <div className="panel-header">
              <div>
                <p className="section-kicker">Parsed Values</p>
                <h2>Structured Fields</h2>
              </div>
            </div>
            <div className="table-shell">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Field</th>
                    <th>Value</th>
                    <th>Method</th>
                    <th>Page</th>
                  </tr>
                </thead>
                <tbody>
                  {result.structured_fields.map((item, index) => (
                    <tr key={`${item.field || 'field'}-${index}`}>
                      <td>{item.field || ''}</td>
                      <td>{item.value || ''}</td>
                      <td>{item.method || ''}</td>
                      <td>{String((item.page_index ?? 0) + 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : null}

        <div className="panel markdown-panel">
          <div className="panel-header">
            <div>
              <p className="section-kicker">Text Output</p>
              <h2>Markdown</h2>
            </div>
            <button
              type="button"
              className="copy-btn"
              onClick={() => copyText('markdown', result.markdown || result.raw_text || 'No text extracted.')}
            >
              {copiedKey === 'markdown' ? 'Copied!' : 'Copy'}
            </button>
          </div>
          <pre className="code-block">{result.markdown || result.raw_text || 'No text extracted.'}</pre>
        </div>

        <div className="panel">
          <div className="panel-header">
            <div>
              <p className="section-kicker">Debug View</p>
              <h2>Raw JSON</h2>
            </div>
            <button
              type="button"
              className="copy-btn"
              onClick={() => copyText('json', JSON.stringify(result, null, 2))}
            >
              {copiedKey === 'json' ? 'Copied!' : 'Copy'}
            </button>
          </div>
          <pre className="code-block">{JSON.stringify(result, null, 2)}</pre>
        </div>
      </section>
    </main>
  );
}

function StatCard({ label, value }) {
  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <strong className="stat-value">{value}</strong>
    </div>
  );
}

function isSupportedFile(file) {
  if (!file) {
    return false;
  }

  const name = file.name.toLowerCase();
  return file.type.startsWith('image/')
    || file.type === 'application/pdf'
    || name.endsWith('.pdf')
    || name.endsWith('.png')
    || name.endsWith('.jpg')
    || name.endsWith('.jpeg')
    || name.endsWith('.tif')
    || name.endsWith('.tiff')
    || name.endsWith('.bmp');
}

function isPdfFile(file) {
  return file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
}

function getDroppedFiles(dataTransfer) {
  if (!dataTransfer) {
    return [];
  }

  if (dataTransfer.items?.length) {
    return Array.from(dataTransfer.items)
      .filter((item) => item.kind === 'file')
      .map((item) => item.getAsFile())
      .filter(Boolean);
  }

  return Array.from(dataTransfer.files || []);
}

function getFileMeta(files) {
  if (!files.length) {
    return 'No file selected.';
  }

  if (files.length === 1) {
    const [file] = files;
    return `${file.name} • ${formatBytes(file.size)}${file.type ? ` • ${file.type}` : ''}`;
  }

  const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
  return `${files.length} files selected • ${formatBytes(totalBytes)} total • select a row to preview`;
}

function FileIcon({ file }) {
  if (isPdfFile(file)) {
    return (
      <svg className="file-list-icon" viewBox="0 0 20 20" fill="none" aria-hidden="true">
        <rect x="3" y="1" width="11" height="15" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
        <path d="M11 1v4h4" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
        <path d="M6 9h5M6 12h3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <rect x="10" y="12" width="7" height="5" rx="1" fill="var(--accent)" />
        <text x="13.5" y="16.2" fontSize="3.8" fill="white" textAnchor="middle" fontWeight="700">PDF</text>
      </svg>
    );
  }
  return (
    <svg className="file-list-icon" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <rect x="2" y="2" width="16" height="16" rx="2" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="7.5" cy="7.5" r="1.5" stroke="currentColor" strokeWidth="1.2" />
      <path d="M2 13l4-3.5 3 2.5 3-4 6 5" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '0 B';
  }

  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatDuration(milliseconds) {
  const totalSeconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

function mergeResults(results) {
  const normalizedResults = results.map(normalizeRunResult);
  const documents = reindexDocuments(normalizedResults.flatMap((result) => result.documents || []));
  const normalizedDocuments = documents
    .map((document) => document.normalized_document)
    .filter(Boolean);

  if (normalizedResults.length === 1) {
    return {
      ...reindexResultDocuments(normalizedResults[0], documents),
      documents,
      normalized_case: mergeNormalizedDocuments(normalizedDocuments),
    };
  }

  const markdownParts = normalizedResults.map((result) => {
    const source = result.source || '';
    const markdown = result.markdown || result.raw_text || '';
    return source ? `### ${source}\n\n${markdown}` : markdown;
  });

  const firstProcessing = normalizedResults[0]?.processing || null;
  const mergedProcessing = firstProcessing
    ? {
        ...firstProcessing,
        downscaled_file_count: normalizedResults.filter((r) => r.processing?.downscaled).length,
      }
    : null;

  return {
    source: normalizedResults.map((result) => result.source).join(', '),
    summary: {
      engine: 'PP-StructureV3',
      page_count: normalizedResults.reduce((sum, result) => sum + (result.summary?.page_count ?? 0), 0),
      table_count: normalizedResults.reduce((sum, result) => sum + (result.summary?.table_count ?? 0), 0),
      field_count: normalizedResults.reduce((sum, result) => sum + (result.summary?.field_count ?? 0), 0),
      text_characters: normalizedResults.reduce((sum, result) => sum + (result.summary?.text_characters ?? 0), 0),
    },
    markdown: markdownParts.join('\n\n---\n\n'),
    raw_text: markdownParts.join('\n\n---\n\n'),
    structured_fields: normalizedResults.flatMap((result) => result.structured_fields || []),
    tables: normalizedResults.flatMap((result) => result.tables || []),
    pages: normalizedResults.flatMap((result) => result.pages || []),
    processing: mergedProcessing,
    documents,
    normalized_case: mergeNormalizedDocuments(normalizedDocuments),
  };
}

function normalizeRunResult(result) {
  const normalizedDocuments = Array.isArray(result.documents) && result.documents.length
    ? result.documents.map(normalizeDocument)
    : [normalizeDocument(result)];

  return {
    ...result,
    summary: normalizeSummary(result.summary),
    structured_fields: Array.isArray(result.structured_fields) ? result.structured_fields : [],
    tables: Array.isArray(result.tables) ? result.tables : [],
    pages: Array.isArray(result.pages) ? result.pages : [],
    markdown: result.markdown || result.raw_text || '',
    raw_text: result.raw_text || result.markdown || '',
    raw_document: normalizeRawDocument(result.raw_document, result),
    normalized_document: normalizeNormalizedDocument(result.normalized_document, result),
    documents: normalizedDocuments,
  };
}

function normalizeDocument(document) {
  return {
    source: document.source || 'Untitled source',
    summary: normalizeSummary(document.summary),
    structured_fields: Array.isArray(document.structured_fields) ? document.structured_fields : [],
    tables: Array.isArray(document.tables) ? document.tables : [],
    pages: Array.isArray(document.pages) ? document.pages : [],
    markdown: document.markdown || document.raw_text || '',
    raw_text: document.raw_text || document.markdown || '',
    processing: document.processing || null,
    raw_document: normalizeRawDocument(document.raw_document, document),
    normalized_document: normalizeNormalizedDocument(document.normalized_document, document),
  };
}

function normalizeSummary(summary) {
  return {
    ...EMPTY_SUMMARY,
    ...(summary || {}),
  };
}

function normalizeNormalizedDocument(normalizedDocument, document) {
  if (!normalizedDocument || typeof normalizedDocument !== 'object') {
    return null;
  }

  const metadata = normalizedDocument.metadata || {};
  return {
    image_id: normalizedDocument.image_id || getImageId(document.source),
    metadata: {
      filename: metadata.filename || document.source || 'Untitled source',
      page_number: metadata.page_number ?? 1,
      ocr_mode: metadata.ocr_mode || document.processing?.ocr_mode_label || document.processing?.ocr_mode || '',
      run_timestamp: metadata.run_timestamp || document.processing?.run_timestamp || '',
    },
    document_context: normalizeKeyFacts(normalizedDocument.document_context),
    key_facts: normalizeKeyFacts(normalizedDocument.key_facts),
    tables: normalizeNormalizedTables(normalizedDocument.tables),
    ocr_text: Array.isArray(normalizedDocument.ocr_text)
      ? normalizedDocument.ocr_text.filter(Boolean)
      : [],
    filtered_noise: Array.isArray(normalizedDocument.filtered_noise)
      ? normalizedDocument.filtered_noise.filter(Boolean)
      : [],
  };
}

function normalizeRawDocument(rawDocument, document) {
  if (!rawDocument || typeof rawDocument !== 'object') {
    return null;
  }

  const metadata = rawDocument.metadata || {};
  return {
    image_id: rawDocument.image_id || getImageId(document.source),
    metadata: {
      filename: metadata.filename || document.source || 'Untitled source',
      page_number: metadata.page_number ?? 1,
      ocr_mode: metadata.ocr_mode || document.processing?.ocr_mode_label || document.processing?.ocr_mode || '',
      run_timestamp: metadata.run_timestamp || document.processing?.run_timestamp || '',
    },
    blocks: Array.isArray(rawDocument.blocks) ? rawDocument.blocks.map((block) => normalizeRawBlock(block)) : [],
  };
}

function normalizeRawBlock(block) {
  const type = block?.type || 'text';
  if (type === 'table') {
    return {
      type: 'table',
      title: block?.title || 'Table',
      headers: Array.isArray(block?.headers) ? block.headers : [],
      rows: Array.isArray(block?.rows) ? block.rows : [],
    };
  }
  return {
    type,
    text: block?.text || '',
  };
}

function normalizeKeyFacts(keyFacts) {
  const facts = {};
  Object.entries(keyFacts || {}).forEach(([key, value]) => {
    if (key && value) {
      facts[key] = value;
    }
  });
  return facts;
}

function normalizeNormalizedTables(tables) {
  return Array.isArray(tables) ? tables
    .map((table) => ({
      table_id: table?.table_id || '',
      title: table?.title || 'Table',
      columns: Array.isArray(table?.columns) ? table.columns : [],
      rows: Array.isArray(table?.rows) ? table.rows : [],
    }))
    .filter((table) => table.columns.length > 0 || table.rows.length > 0) : [];
}

function mergeNormalizedDocuments(documents) {
  if (!documents.length) {
    return null;
  }

  const structuredDocuments = documents.filter(isStructuredNormalizedDocument);
  const firstMetadata = documents[0]?.metadata || {};

  const documentContext = {};
  structuredDocuments.forEach((document) => {
    Object.entries(document.document_context || {}).forEach(([key, value]) => {
      if (!documentContext[key] && value) {
        documentContext[key] = value;
      }
    });
  });

  const keyFacts = {};
  structuredDocuments.forEach((document) => {
    Object.entries(document.key_facts || {}).forEach(([key, value]) => {
      if (!keyFacts[key] && value) {
        keyFacts[key] = value;
      }
    });
  });

  const tables = [];
  const seenTables = new Set();
  structuredDocuments.forEach((document) => {
    (document.tables || []).forEach((table) => {
      const signature = JSON.stringify({
        table_id: table.table_id,
        title: table.title,
        columns: table.columns,
        rows: table.rows,
      });
      if (seenTables.has(signature)) {
        return;
      }
      seenTables.add(signature);
      tables.push(table);
    });
  });

  const ocrText = dedupeLines(documents.flatMap((document) => document.ocr_text || []));
  const filteredNoise = dedupeLines(structuredDocuments.flatMap((document) => document.filtered_noise || []));

  return {
    metadata: {
      filenames: documents.map((document) => document.metadata?.filename).filter(Boolean),
      page_count: documents.length,
      ocr_mode: firstMetadata.ocr_mode || '',
      run_timestamp: firstMetadata.run_timestamp || '',
    },
    image_ids: documents.map((document) => document.image_id).filter(Boolean),
    document_context: documentContext,
    key_facts: keyFacts,
    tables,
    ocr_text: ocrText,
    filtered_noise: filteredNoise,
  };
}

function isStructuredNormalizedDocument(document) {
  return Boolean(
    Object.keys(document?.document_context || {}).length
    || Object.keys(document?.key_facts || {}).length
    || (document?.tables || []).length,
  );
}

function dedupeLines(lines) {
  const seen = new Set();
  return (lines || []).filter((line) => {
    if (!line) {
      return false;
    }
    const signature = String(line).trim().toLowerCase();
    if (!signature || seen.has(signature)) {
      return false;
    }
    seen.add(signature);
    return true;
  });
}

function reindexDocuments(documents) {
  return documents.map((document, index) => {
    const pageNumber = index + 1;
    return {
      ...document,
      raw_document: reindexRawDocument(document.raw_document, document, pageNumber),
      normalized_document: reindexNormalizedDocument(document.normalized_document, document, pageNumber),
    };
  });
}

function reindexResultDocuments(result, documents) {
  if (!documents.length) {
    return result;
  }

  return {
    ...result,
    raw_document: reindexRawDocument(result.raw_document, result, 1),
    normalized_document: reindexNormalizedDocument(result.normalized_document, result, 1),
  };
}

function reindexRawDocument(rawDocument, document, pageNumber) {
  if (!rawDocument) {
    return null;
  }
  return {
    ...rawDocument,
    image_id: rawDocument.image_id || getImageId(document.source),
    metadata: {
      ...(rawDocument.metadata || {}),
      filename: rawDocument.metadata?.filename || document.source || 'Untitled source',
      page_number: pageNumber,
      ocr_mode: rawDocument.metadata?.ocr_mode || document.processing?.ocr_mode_label || document.processing?.ocr_mode || '',
      run_timestamp: rawDocument.metadata?.run_timestamp || document.processing?.run_timestamp || '',
    },
  };
}

function reindexNormalizedDocument(normalizedDocument, document, pageNumber) {
  if (!normalizedDocument) {
    return null;
  }
  return {
    ...normalizedDocument,
    image_id: normalizedDocument.image_id || getImageId(document.source),
    metadata: {
      ...(normalizedDocument.metadata || {}),
      filename: normalizedDocument.metadata?.filename || document.source || 'Untitled source',
      page_number: pageNumber,
      ocr_mode: normalizedDocument.metadata?.ocr_mode || document.processing?.ocr_mode_label || document.processing?.ocr_mode || '',
      run_timestamp: normalizedDocument.metadata?.run_timestamp || document.processing?.run_timestamp || '',
    },
  };
}

function getImageId(source) {
  const filename = String(source || 'document');
  return filename.replace(/^.*[\\/]/, '').replace(/\.[^.]+$/, '');
}

function getDocumentEntries(result) {
  if (Array.isArray(result.documents) && result.documents.length) {
    return result.documents;
  }

  if (result?.source || result?.markdown || result?.raw_text) {
    return [normalizeDocument(result)];
  }

  return [];
}

function matchFilesToDocuments(files, documents) {
  const nameBuckets = new Map();
  const usedIndexes = new Set();

  files.forEach((file, index) => {
    const bucket = nameBuckets.get(file.name) || [];
    bucket.push(index);
    nameBuckets.set(file.name, bucket);
  });

  return documents.map((document) => {
    const bucket = nameBuckets.get(document.source) || [];
    while (bucket.length > 0) {
      const index = bucket.shift();
      if (!usedIndexes.has(index)) {
        usedIndexes.add(index);
        return files[index];
      }
    }

    const fallbackIndex = files.findIndex((_, index) => !usedIndexes.has(index));
    if (fallbackIndex >= 0) {
      usedIndexes.add(fallbackIndex);
      return files[fallbackIndex];
    }

    return null;
  });
}

async function buildReportPdf({ documents, files, summary, lastRunMs, modeLabel }) {
  const pdf = new jsPDF({
    orientation: 'portrait',
    unit: 'pt',
    format: 'a4',
    compress: true,
  });
  const pageWidth = pdf.internal.pageSize.getWidth();
  const pageHeight = pdf.internal.pageSize.getHeight();
  const margin = 42;
  const contentWidth = pageWidth - (margin * 2);
  const generatedAt = new Date().toLocaleString();

  drawReportCover(pdf, {
    margin,
    contentWidth,
    summary,
    lastRunMs,
    generatedAt,
    modeLabel,
    documentCount: documents.length,
  });

  for (let index = 0; index < documents.length; index += 1) {
    pdf.addPage();
    await drawDocumentPage(pdf, {
      document: documents[index],
      file: files[index] || null,
      index,
      total: documents.length,
      margin,
      contentWidth,
      pageHeight,
    });
  }

  return pdf.output('blob');
}

function drawReportCover(pdf, { margin, contentWidth, summary, lastRunMs, generatedAt, modeLabel, documentCount }) {
  let y = margin + 12;

  pdf.setFillColor(244, 241, 232);
  pdf.roundedRect(margin, y, contentWidth, 136, 22, 22, 'F');

  pdf.setTextColor(15, 107, 111);
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(11);
  pdf.text('OCR DEMO REPORT', margin + 22, y + 24);

  pdf.setTextColor(22, 34, 59);
  pdf.setFontSize(25);
  pdf.text('Structured Extraction Summary', margin + 22, y + 56);

  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(11);
  pdf.setTextColor(89, 101, 125);
  pdf.text(`Generated ${generatedAt}`, margin + 22, y + 82);
  pdf.text(`Mode ${modeLabel}`, margin + 22, y + 100);
  pdf.text(`Files ${documentCount}`, margin + 22, y + 118);
  if (lastRunMs > 0) {
    pdf.text(`Run time ${formatDuration(lastRunMs)}`, margin + 160, y + 100);
  }

  y += 164;

  const stats = [
    ['Engine', summary.engine || 'PP-StructureV3'],
    ['Pages', String(summary.page_count ?? 0)],
    ['Tables', String(summary.table_count ?? 0)],
    ['Fields', String(summary.field_count ?? 0)],
    ['Text Chars', String(summary.text_characters ?? 0)],
  ];

  stats.forEach(([label, value], index) => {
    const columnWidth = (contentWidth - 16) / 2;
    const cardX = margin + ((index % 2) * (columnWidth + 16));
    const cardY = y + (Math.floor(index / 2) * 88);

    pdf.setFillColor(255, 255, 255);
    pdf.setDrawColor(221, 229, 242);
    pdf.roundedRect(cardX, cardY, columnWidth, 72, 18, 18, 'FD');

    pdf.setTextColor(89, 101, 125);
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(9);
    pdf.text(label.toUpperCase(), cardX + 16, cardY + 22);

    pdf.setTextColor(22, 34, 59);
    pdf.setFontSize(19);
    pdf.text(value, cardX + 16, cardY + 50);
  });

  y += 264;

  pdf.setTextColor(22, 34, 59);
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(13);
  pdf.text('Report Layout', margin, y);

  const layoutCopy = [
    'Each following page keeps the source screenshot close to a cleaned, readable OCR summary.',
    'Structured fields are shown first when available, followed by table highlights and readable page text.',
  ];
  y += 20;
  drawWrappedParagraphs(pdf, layoutCopy, {
    x: margin,
    y,
    maxWidth: contentWidth,
    pageHeight: pdf.internal.pageSize.getHeight(),
    margin,
  });
}

async function drawDocumentPage(pdf, { document, file, index, total, margin, contentWidth, pageHeight }) {
  let y = margin;

  pdf.setTextColor(15, 107, 111);
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(10);
  pdf.text(`SOURCE ${index + 1} OF ${total}`, margin, y);

  y += 22;
  pdf.setTextColor(22, 34, 59);
  pdf.setFontSize(20);
  pdf.text(document.source || `Document ${index + 1}`, margin, y);

  y += 18;
  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(10.5);
  pdf.setTextColor(89, 101, 125);
  pdf.text(buildDocumentMetaLine(document), margin, y);

  y += 24;

  if (file && !isPdfFile(file)) {
    try {
      const image = await prepareImageForPdf(file);
      const imageBoxHeight = 248;
      const imageScale = Math.min(contentWidth / image.width, imageBoxHeight / image.height, 1);
      const drawWidth = image.width * imageScale;
      const drawHeight = image.height * imageScale;
      const drawX = margin + ((contentWidth - drawWidth) / 2);

      pdf.setFillColor(255, 255, 255);
      pdf.setDrawColor(221, 229, 242);
      pdf.roundedRect(margin, y, contentWidth, drawHeight + 24, 18, 18, 'FD');
      pdf.addImage(image.dataUrl, image.format, drawX, y + 12, drawWidth, drawHeight, undefined, 'FAST');
      y += drawHeight + 40;
    } catch {
      y = drawReportPlaceholder(pdf, {
        x: margin,
        y,
        width: contentWidth,
        height: 116,
        title: 'Image preview unavailable',
        subtitle: 'The OCR summary below is still included in this report.',
      });
    }
  } else {
    y = drawReportPlaceholder(pdf, {
      x: margin,
      y,
      width: contentWidth,
      height: 116,
      title: file ? 'PDF source attached to OCR run' : 'Source preview unavailable',
      subtitle: file ? `${file.name} • preview omitted from in-browser report export` : 'Only OCR text is available for this page.',
    });
  }

  const sections = buildReadableSections(document);

  sections.forEach((section) => {
    y = ensurePdfRoom(pdf, y, 18, { pageHeight, margin });

    pdf.setTextColor(22, 34, 59);
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(12.5);
    pdf.text(section.title, margin, y);
    y += 18;

    if (section.type === 'table') {
      y = drawPdfTable(pdf, {
        title: section.title,
        rows: section.rows,
        headerRowCount: section.headerRowCount || 0,
        x: margin,
        y,
        width: contentWidth,
        pageHeight,
        margin,
      });
      y += 8;
      return;
    }

    pdf.setFont('helvetica', 'normal');
    pdf.setFontSize(10.5);
    pdf.setTextColor(42, 52, 73);
    y = drawWrappedParagraphs(pdf, section.lines, {
      x: margin,
      y,
      maxWidth: contentWidth,
      pageHeight,
      margin,
    });
    y += 6;
  });
}

function drawReportPlaceholder(pdf, { x, y, width, height, title, subtitle }) {
  pdf.setFillColor(249, 251, 255);
  pdf.setDrawColor(221, 229, 242);
  pdf.roundedRect(x, y, width, height, 18, 18, 'FD');

  pdf.setTextColor(22, 34, 59);
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(13);
  pdf.text(title, x + 18, y + 36);

  pdf.setTextColor(89, 101, 125);
  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(10.5);
  const wrapped = pdf.splitTextToSize(subtitle, width - 36);
  pdf.text(wrapped, x + 18, y + 58);

  return y + height + 18;
}

function ensurePdfRoom(pdf, y, neededHeight, { pageHeight, margin }) {
  if (y + neededHeight <= pageHeight - margin) {
    return y;
  }

  pdf.addPage();
  return margin;
}

function drawWrappedParagraphs(pdf, lines, { x, y, maxWidth, pageHeight, margin }) {
  const paragraphs = lines.filter(Boolean);

  paragraphs.forEach((line) => {
    const wrapped = pdf.splitTextToSize(line, maxWidth);
    wrapped.forEach((segment) => {
      if (y > pageHeight - margin) {
        pdf.addPage();
        y = margin;
      }
      pdf.text(segment, x, y);
      y += 14;
    });
    y += 6;
  });

  return y;
}

function drawPdfTable(pdf, {
  title,
  rows,
  headerRowCount = 0,
  x,
  y,
  width,
  pageHeight,
  margin,
}) {
  const normalizedRows = normalizePdfTableRows(rows);
  if (!normalizedRows.length) {
    return y;
  }

  const columnCount = Math.max(...normalizedRows.map((row) => row.length), 1);
  const tableRows = normalizedRows.map((row) => Array.from({ length: columnCount }, (_, index) => normalizeWhitespace(row[index] || '')));
  const columnWidths = getPdfColumnWidths(tableRows, width);
  const fontSize = columnCount >= 8 ? 7 : columnCount >= 6 ? 7.6 : 8.5;
  const lineHeight = fontSize * 1.34;
  const cellPaddingX = 6;
  const cellPaddingY = 5;
  const wrappedRows = tableRows.map((row) => row.map((cell, columnIndex) => {
    const wrapped = pdf.splitTextToSize(cell || ' ', Math.max(columnWidths[columnIndex] - (cellPaddingX * 2), 16));
    return clampWrappedLines(wrapped, 6);
  }));
  const headerRows = wrappedRows.slice(0, headerRowCount);
  const bodyRows = wrappedRows.slice(headerRowCount);

  pdf.setFontSize(fontSize);

  const renderRow = (wrappedCells, rowY, isHeader) => {
    const rowHeight = Math.max(...wrappedCells.map((lines) => Math.max(lines.length, 1))) * lineHeight + (cellPaddingY * 2);
    let cursorX = x;

    wrappedCells.forEach((lines, columnIndex) => {
      const cellWidth = columnWidths[columnIndex];
      pdf.setDrawColor(221, 229, 242);
      if (isHeader) {
        pdf.setFillColor(240, 244, 252);
      } else {
        pdf.setFillColor(255, 255, 255);
      }
      pdf.rect(cursorX, rowY, cellWidth, rowHeight, 'FD');

      pdf.setTextColor(isHeader ? 89 : 42, isHeader ? 101 : 52, isHeader ? 125 : 73);
      pdf.setFont('helvetica', isHeader ? 'bold' : 'normal');
      pdf.text(lines, cursorX + cellPaddingX, rowY + cellPaddingY + fontSize, {
        baseline: 'alphabetic',
      });

      cursorX += cellWidth;
    });

    return rowHeight;
  };

  const renderHeaderRows = () => {
    headerRows.forEach((wrappedCells) => {
      const rowHeight = Math.max(...wrappedCells.map((lines) => Math.max(lines.length, 1))) * lineHeight + (cellPaddingY * 2);
      y = ensurePdfRoom(pdf, y, rowHeight, { pageHeight, margin });
      renderRow(wrappedCells, y, true);
      y += rowHeight;
    });
  };

  renderHeaderRows();

  bodyRows.forEach((wrappedCells) => {
    const rowHeight = Math.max(...wrappedCells.map((lines) => Math.max(lines.length, 1))) * lineHeight + (cellPaddingY * 2);

    if (y + rowHeight > pageHeight - margin) {
      pdf.addPage();
      y = margin;

      pdf.setTextColor(22, 34, 59);
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(12.5);
      pdf.text(`${title} (cont.)`, x, y);
      y += 18;

      renderHeaderRows();
    }

    renderRow(wrappedCells, y, false);
    y += rowHeight;
  });

  return y;
}

function normalizePdfTableRows(rows) {
  return (rows || [])
    .map((row) => Array.isArray(row)
      ? row.map((cell) => normalizeWhitespace(cell || ''))
      : [normalizeWhitespace(row || '')])
    .filter((row) => row.some(Boolean));
}

function getPdfColumnWidths(rows, totalWidth) {
  const columnCount = Math.max(...rows.map((row) => row.length), 1);
  const weights = Array.from({ length: columnCount }, () => 1);

  rows.forEach((row) => {
    row.forEach((cell, index) => {
      weights[index] = Math.max(weights[index], Math.min((cell || '').length, 36) || 1);
    });
  });

  const totalWeight = weights.reduce((sum, value) => sum + value, 0);
  const baseWidths = weights.map((weight) => (weight / totalWeight) * totalWidth);
  const minWidth = Math.max(46, totalWidth / (columnCount * 1.7));
  let widths = baseWidths.map((width) => Math.max(width, minWidth));
  const widthSum = widths.reduce((sum, value) => sum + value, 0);

  if (widthSum > totalWidth) {
    const scale = totalWidth / widthSum;
    widths = widths.map((width) => width * scale);
  } else if (widthSum < totalWidth) {
    widths[widths.length - 1] += totalWidth - widthSum;
  }

  return widths;
}

function clampWrappedLines(lines, maxLines) {
  if (lines.length <= maxLines) {
    return lines;
  }

  const visibleLines = lines.slice(0, maxLines);
  visibleLines[maxLines - 1] = `${visibleLines[maxLines - 1].replace(/\s+$/, '')}…`;
  return visibleLines;
}

async function prepareImageForPdf(file) {
  const dataUrl = await readFileAsDataUrl(file);
  const image = await loadImage(dataUrl);
  const canvas = document.createElement('canvas');
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Unable to prepare the image for PDF export.');
  }

  context.drawImage(image, 0, 0, width, height);
  const format = getPdfImageFormat(file);

  return {
    dataUrl: format === 'JPEG' ? canvas.toDataURL('image/jpeg', 0.92) : canvas.toDataURL('image/png'),
    width,
    height,
    format,
  };
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error(`Unable to read ${file.name}.`));
    reader.readAsDataURL(file);
  });
}

function loadImage(dataUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Unable to load the uploaded image for the PDF report.'));
    image.src = dataUrl;
  });
}

function getPdfImageFormat(file) {
  return file.type === 'image/jpeg' || file.type === 'image/jpg' ? 'JPEG' : 'PNG';
}

function buildDocumentMetaLine(document) {
  const summary = document.summary || EMPTY_SUMMARY;
  return [
    `Pages ${summary.page_count ?? 0}`,
    `Tables ${summary.table_count ?? 0}`,
    `Fields ${summary.field_count ?? 0}`,
    `Text ${summary.text_characters ?? 0} chars`,
  ].join(' • ');
}

function buildReadableSections(document) {
  const sections = [];
  const fieldRows = getFieldRows(document.structured_fields);
  const tableSections = getTableSections(document.tables);
  const textLines = getReadableTextLines(document);

  if (fieldRows.length > 0) {
    sections.push({
      title: 'Structured Fields',
      type: 'table',
      headerRowCount: 1,
      rows: [
        ['Field', 'Value'],
        ...fieldRows,
      ],
    });
  }

  tableSections.forEach((section) => {
    sections.push(section);
  });

  if (textLines.length > 0) {
    sections.push({
      title: 'Readable OCR',
      lines: textLines,
    });
  }

  if (!sections.length) {
    sections.push({
      title: 'Readable OCR',
      lines: ['No text was extracted for this source. The original screenshot above is included for context.'],
    });
  }

  return sections;
}

function getFieldRows(fields) {
  const seen = new Set();

  return (fields || [])
    .map((item) => {
      const field = normalizeWhitespace(item.field || '');
      const value = normalizeWhitespace(item.value || '');
      if (!field && !value) {
        return null;
      }
      return [field || 'Field', value || ''];
    })
    .filter((row) => {
      const signature = JSON.stringify(row);
      if (!row || seen.has(signature)) {
        return false;
      }
      seen.add(signature);
      return true;
    })
    .slice(0, 14);
}

function getTableSections(tables) {
  return dedupeTables(tables).slice(0, 3).map((table, index) => {
    const rows = Array.isArray(table.rows) ? table.rows : [];
    const cells = Array.isArray(table.cells) ? table.cells : [];

    if (cells.length > 0) {
      const dataRows = cells
        .map((cell) => {
          const field = normalizeWhitespace(cell.field || '');
          const value = normalizeWhitespace(cell.value || '');
          if (!field && !value) {
            return null;
          }
          return [field || 'Field', value || ''];
        })
        .filter(Boolean)
        .slice(0, 12);

      return {
        title: `Table ${index + 1}`,
        type: 'table',
        headerRowCount: 1,
        rows: [
          ['Field', 'Value'],
          ...dataRows,
        ],
      };
    }

    const tableRows = rows
      .map((row) => row.map((cell) => normalizeWhitespace(cell)))
      .filter((row) => row.some(Boolean))
      .slice(0, 10);

    if (rows.length > 10) {
      tableRows.push([`${rows.length - 10} more row(s) not shown`]);
    }

    return {
      title: `Table ${index + 1}`,
      type: 'table',
      headerRowCount: 0,
      rows: tableRows,
    };
  }).filter((section) => section.rows.length > 0);
}

function dedupeTables(tables) {
  const seen = new Set();

  return (tables || []).filter((table) => {
    const rows = Array.isArray(table.rows) ? table.rows : [];
    const cells = Array.isArray(table.cells) ? table.cells : [];
    const signature = JSON.stringify({
      rows: rows.slice(0, 6),
      cells: cells.slice(0, 12),
    });

    if (seen.has(signature)) {
      return false;
    }

    seen.add(signature);
    return rows.length > 0 || cells.length > 0;
  });
}

function getReadableTextLines(document) {
  const sources = Array.isArray(document.pages) && document.pages.length > 0
    ? document.pages.map((page) => page.markdown || '')
    : [document.markdown || document.raw_text || ''];

  const seen = new Set();

  return sources
    .flatMap((source) => extractReadableLines(source))
    .filter((line) => {
      if (!line || seen.has(line)) {
        return false;
      }
      seen.add(line);
      return true;
    })
    .slice(0, 18);
}

function extractReadableLines(source) {
  if (!source) {
    return [];
  }

  return source
    .split(/\n{2,}/)
    .flatMap((block) => {
      const trimmed = block.trim();
      if (!trimmed || trimmed === '---') {
        return [];
      }

      if (/<table[\s>]/i.test(trimmed)) {
        return [];
      }

      if (/<[a-z][\s\S]*>/i.test(trimmed)) {
        return extractLinesFromHtml(trimmed);
      }

      const text = normalizeWhitespace(trimmed.replace(/^#+\s*/, ''));
      return text ? [text] : [];
    });
}

function extractLinesFromHtml(html) {
  const parser = new DOMParser();
  const parsed = parser.parseFromString(`<body>${html}</body>`, 'text/html');
  const rows = Array.from(parsed.querySelectorAll('tr'));

  if (rows.length > 0) {
    return rows
      .map((row) => Array.from(row.querySelectorAll('th,td'))
        .map((cell) => normalizeWhitespace(cell.textContent || ''))
        .filter(Boolean)
        .join(' | '))
      .filter(Boolean);
  }

  const text = normalizeWhitespace(parsed.body.textContent || '');
  return text ? [text] : [];
}

function normalizeWhitespace(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}
