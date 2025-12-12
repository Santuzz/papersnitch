
function interpolateColor(value) {
    value = Math.max(0, Math.min(100, value));
    const t = value / 100;

    // Hue 0..270 (rosso → violetto)
    const hue = t * 270;

    // Toni meno accesi: saturazione e luminosità moderate
    const saturation = 0.55; // 0.5–0.6 produce tonalità più vicine ai tuoi colori
    const brightness = 0.70;

    function hsvToRgb(h, s, v) {
        const c = v * s;
        const hh = (h / 60) % 6;
        const x = c * (1 - Math.abs((hh % 2) - 1));
        let r = 0, g = 0, b = 0;

        if (0 <= hh && hh < 1) { r = c; g = x; b = 0; }
        else if (1 <= hh && hh < 2) { r = x; g = c; b = 0; }
        else if (2 <= hh && hh < 3) { r = 0; g = c; b = x; }
        else if (3 <= hh && hh < 4) { r = 0; g = x; b = c; }
        else if (4 <= hh && hh < 5) { r = x; g = 0; b = c; }
        else if (5 <= hh && hh < 6) { r = c; g = 0; b = x; }

        const m = v - c;
        return {
            r: Math.round((r + m) * 255),
            g: Math.round((g + m) * 255),
            b: Math.round((b + m) * 255)
        };
    }

    function rgbToHex({ r, g, b }) {
        return (
            "#" +
            r.toString(16).padStart(2, "0") +
            g.toString(16).padStart(2, "0") +
            b.toString(16).padStart(2, "0")
        );
    }

    return rgbToHex(hsvToRgb(hue, saturation, brightness));
}

function loadAnalysisDetails(analysisId) {
    const modalContent = document.getElementById('analysis-modal-content');
    modalContent.innerHTML = `
            <div class="text-center py-5">
                <i class="fas fa-spinner fa-spin fa-2x"></i>
                <p class="mt-2">Loading analysis...</p>
            </div>
        `;

    fetch(`/analysis/${analysisId}/`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                modalContent.innerHTML = `
                        <div class="alert alert-danger">
                            <strong>Error:</strong> ${data.error}
                        </div>
                    `;
                return;
            }

            modalContent.innerHTML = renderAnalysisDetails(data, true);
        })
        .catch(error => {
            modalContent.innerHTML = `
                    <div class="alert alert-danger">
                        <strong>Error:</strong> Failed to load analysis details.
                    </div>
                `;
        });
}

function renderAnalysisDetails(data, is_details = true) {
    const result = data.result;
    const uniqueId = Math.random().toString(36).slice(2, 11);

    let html_start = `
            <div class="mb-3">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                    `;
    let html_end = `
                            <i class="fas fa-clock mr-1"></i>${data.duration}s |
                            <i class="fas fa-arrow-down mr-1"></i>Input: ${data.input_tokens} tokens |
                            <i class="fas fa-arrow-up mr-1"></i>Output: ${data.output_tokens} tokens
                        </div>
                    </div>
                    <div class="text-right">
                        <div class="h4 mb-0 text-primary font-weight-bold">
                            ${data.final_score !== undefined ? data.final_score : 'N/A'}
                        </div>
                        <small class="text-muted">Final Score</small>
                    </div>
                </div>
            </div>
        `;
    let html_mid = ``;
    //For details 
    if (is_details) {
        html_mid = `
                        <h6 class="text-primary"><i class="fas fa-file-alt mr-2"></i>${data.paper_title}</h6>
                        <div class="text-muted small">
                            <span class="badge badge-info mr-2">${data.model_name}</span>
                            <i class="fas fa-calendar mr-1"></i>${data.created_at} |
                            `;
    }
    else {
        // For analysis
        html_mid = `
                        <div class="text-muted small">
                            `;
    }

    let html = html_start + html_mid + html_end;
    //For details 
    if (is_details) {
        if (data.error) {
            html += `
                <div class="alert alert-danger">
                    <strong>Error:</strong> ${data.error}
                </div>
            `;
            return html;
        }
    }

    if (is_details) {
        if (!result) {
            html += `<div class="alert alert-warning">No analysis data available.</div>`;
            return html;
        }
    }

    // Datasets section
    html += `
            <div class="card criterion-card">
                <div class="card-header bg-light collapsible" data-toggle="collapse" data-target="#datasets-${uniqueId}" aria-expanded="true">
                    <div class="d-flex justify-content-between align-items-center">
                        <h6 class="mb-0"><i class="fas fa-database mr-2"></i>Datasets</h6>
                        <i class="fas fa-chevron-down toggle-icon"></i>
                    </div>
                </div>
                <div id="datasets-${uniqueId}" class="collapse show">
                    <div class="card-body">
                        ${result.datasets && result.datasets.extracted && result.datasets.extracted.length > 0
            ? `<ul class="dataset-list mb-0">${result.datasets.extracted.map(d => `<li>${d}</li>`).join('')}</ul>`
            : '<p class="text-muted mb-0">No datasets identified</p>'}
                    </div>
                </div>
            </div>
        `;

    // Scored sections
    const scoredSections = [
        { key: 'code', icon: 'fa-code', title: 'Code Repository' },
        { key: 'annotation', icon: 'fa-tags', title: 'Annotation' },
        { key: 'preprocessing', icon: 'fa-cogs', title: 'Preprocessing' },
        { key: 'evaluation', icon: 'fa-chart-bar', title: 'Evaluation' },
        { key: 'licensing', icon: 'fa-balance-scale', title: 'Licensing' }
    ];

    for (const section of scoredSections) {
        const data = result[section.key];
        const score = data && typeof data.score === 'number' ? data.score : 0;
        const collapseId = `${section.key}-${uniqueId}`;

        html += `
                <div class="card criterion-card">
                    <div class="card-header bg-light collapsible" data-toggle="collapse" data-target="#${collapseId}" aria-expanded="true">
                        <div class="d-flex justify-content-between align-items-center">
                            <h6 class="mb-0"><i class="fas ${section.icon} mr-2"></i>${section.title}
                            ${section.key === 'code'
                ? `<a href="${data.url}" target="_blank" class="btn btn-sm btn-outline-secondary">
                                       <i class="fas fa-external-link-alt mr-1"></i>View Repository
                                   </a>`
                : ''}</h6>
                            <div class="d-flex align-items-center">
                                ${score === -1
                ? `<span class="score-badge score-${score} mr-2">n/a</span>`
                : `<span class="score-badge score-${score} mr-2">${score}</span>`}
                                <i class="fas fa-chevron-down toggle-icon"></i>
                            </div>
                        </div>
                    </div>
                    <div id="${collapseId}" class="collapse show">
                        <div class="card-body">
                            ${data && data.extracted
                ? `
                                    <h6 class="text-muted">Extracted Information:</h6>
                                    <div class="extracted-text">${data.extracted}</div>
                                    ${data.score_explanation
                    ? `<h6 class="text-muted mt-3">Score Explanation:</h6><p class="mb-0">${data.score_explanation}</p>`
                    : ''}
                                  `
                : '<p class="text-muted mb-0">No information extracted for this criterion</p>'}
                        </div>
                    </div>
                </div>
            `;
    }

    return html;
}


