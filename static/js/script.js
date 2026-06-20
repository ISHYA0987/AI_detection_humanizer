async function detect() {

    const text = document.getElementById("text").value;

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

    document.getElementById("result").innerHTML = `

<h2>${result.prediction}</h2>

<p>Confidence : ${result.confidence}%</p>

<p>AI : ${result.ai_probability}%</p>

<p>Human : ${result.human_probability}%</p>

`;

}