

async function humanize() {

    const text = document.getElementById("text").value.trim();

    const resultSection = document.getElementById("resultSection");
    const originalText = document.getElementById("originalText");
    const humanizedText = document.getElementById("humanizedText");

    if (!text) {

        alert("Please enter some text.");

        return;

    }


    resultSection.style.display = "block";


    originalText.value = text;

    humanizedText.value = "Humanizing... Please wait.";

    try {

        const response = await fetch("/humanize", {

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

            alert(result.message);

            humanizedText.value = "";

            return;

        }

        humanizedText.value = result.humanized_text;


        resultSection.scrollIntoView({

            behavior: "smooth",
            block: "start"

        });

    }

    catch (error) {

        console.error(error);

        humanizedText.value = "";

        alert("Failed to humanize text.");

    }

}




function copyHumanizedText() {

    const output = document.getElementById("humanizedText");

    if (!output.value.trim()) {

        alert("Nothing to copy.");

        return;

    }

    navigator.clipboard.writeText(output.value);

    alert("Copied Successfully!");

}




function clearText() {

    document.getElementById("text").value = "";

    document.getElementById("originalText").value = "";

    document.getElementById("humanizedText").value = "";

    document.getElementById("resultSection").style.display = "none";

}