async function detect() {

    const text = document.getElementById("text").value.trim();

    const resultDiv = document.getElementById("result");

    if (!text) {

        resultDiv.innerHTML = `
            <div class="result-card">
                <h2 style="color:#f59e0b;">
                    Please enter some text.
                </h2>
            </div>
        `;

        return;
    }

    resultDiv.innerHTML = `
        <div class="result-card">
            <h2>Analyzing...</h2>
            <p>Please wait while the model processes your text.</p>
        </div>
    `;

    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {

                "Content-Type": "application/json"

            },

            body: JSON.stringify({

                text: text

            })

        });

        const result = await response.json();

        if (!result.success) {

            resultDiv.innerHTML = `

                <div class="result-card">

                    <h2 style="color:#fbbf24;">

                        ⚠ ${result.message}

                    </h2>

                </div>

            `;

            return;

        }

        // -------------------------
        // Statistics
        // -------------------------

        const stats = result.statistics;

        // -------------------------
        // Sentence Analysis
        // -------------------------

        let sentenceHTML = "";

        result.sentence_analysis.forEach(item => {

            sentenceHTML += `

                <div class="sentence ${item.color}">

                    <p>

                        ${item.sentence}

                    </p>

                    <span>

                        ${item.prediction}
                        •
                        ${item.confidence.toFixed(2)}%

                    </span>

                </div>

            `;

        });

        // -------------------------
        // Main Result
        // -------------------------

        resultDiv.innerHTML = `

        <div class="result-card">

            <div class="prediction-box">

                <h2>${result.prediction}</h2>

                <span class="confidence">

                    Confidence :
                    ${result.confidence}%

                </span>

            </div>

            <div class="analysis-section">

                <h3>

                    Confidence Gauge

                </h3>

                <div id="gauge-container"></div>

            </div>

            <div class="analysis-section">

                <h3>

                    Probability

                </h3>

                <div class="progress-item">

                    <span>AI</span>

                    <div class="progress">

                        <div class="progress-fill ai"

                             style="width:${result.ai_probability}%">

                        </div>

                    </div>

                    <span>${result.ai_probability}%</span>

                </div>

                <div class="progress-item">

                    <span>Human</span>

                    <div class="progress">

                        <div class="progress-fill human"

                             style="width:${result.human_probability}%">

                        </div>

                    </div>

                    <span>${result.human_probability}%</span>

                </div>

            </div>

            <div class="analysis-section">

                <h3>

                    Text Statistics

                </h3>

                <div class="stats-grid">

                    <div class="stat-card">

                        <h4>Words</h4>

                        <p>${stats.words}</p>

                    </div>

                    <div class="stat-card">

                        <h4>Characters</h4>

                        <p>${stats.characters}</p>

                    </div>

                    <div class="stat-card">

                        <h4>Sentences</h4>

                        <p>${stats.sentences}</p>

                    </div>

                    <div class="stat-card">

                        <h4>Paragraphs</h4>

                        <p>${stats.paragraphs}</p>

                    </div>

                    <div class="stat-card">

                        <h4>Reading Time</h4>

                        <p>${stats.reading_time} min</p>

                    </div>

                    <div class="stat-card">

                        <h4>Avg Sentence</h4>

                        <p>${stats.avg_sentence}</p>

                    </div>

                </div>

            </div>

            <div class="analysis-section">

                <h3>

                    Sentence Analysis

                </h3>

                <div id="sentence-analysis">

                    ${sentenceHTML}

                </div>

            </div>

        </div>

        `;

        // -------------------------
        // Render Gauge
        // -------------------------

        renderGauge(

            "gauge-container",

            result.confidence,

            result.prediction

        );

    }

    catch(error){

        console.error(error);

        resultDiv.innerHTML = `

            <div class="result-card">

                <h2 style="color:#ef4444;">

                    Something went wrong.

                </h2>

            </div>

        `;

    }

}