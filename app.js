fetch("http://localhost:7860")  // Adjust port if necessary
    .then(response => {
        if (response.ok) {
            document.getElementById("gradio-app").innerHTML = "<iframe src='http://localhost:7860/' style='width: 100%; height: 100%; border: none;'></iframe>";
        } else {
            console.error("Failed to load Gradio interface.");
        }
    })
    .catch(error => console.error("Error:", error));
