// ===============================
// AI Confidence Gauge
// ===============================

function createGauge(confidence, prediction) {

    confidence = Math.max(0, Math.min(100, confidence));

    const radius = 90;
    const circumference = 2 * Math.PI * radius;

    const offset =
        circumference - (confidence / 100) * circumference;

    let color = "#22c55e"; // Human (Green)

    if (prediction.toLowerCase() === "ai") {
        color = "#ef4444"; // AI (Red)
    }

    return `
    <div class="gauge-wrapper">

        <svg class="gauge" width="220" height="220">

            <circle
                cx="110"
                cy="110"
                r="${radius}"
                class="gauge-bg">
            </circle>

            <circle
                cx="110"
                cy="110"
                r="${radius}"
                class="gauge-progress"
                stroke="${color}"
                stroke-dasharray="${circumference}"
                stroke-dashoffset="${offset}">
            </circle>

        </svg>

        <div class="gauge-text">

            <h2>${confidence.toFixed(1)}%</h2>

            <p>${prediction}</p>

        </div>

    </div>
    `;
}

// ===============================
// Render Gauge
// ===============================

function renderGauge(containerId, confidence, prediction) {

    const container = document.getElementById(containerId);

    if (!container) return;

    container.innerHTML = createGauge(confidence, prediction);

}