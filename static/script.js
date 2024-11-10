document.getElementById("uploadForm").addEventListener("submit", async (event) => {
    event.preventDefault();

    const imageInput = document.getElementById("imageInput");
    if (imageInput.files.length === 0) {
        alert("Please select an image.");
        return;
    }

    const formData = new FormData();
    formData.append("file", imageInput.files[0]);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });
        const result = await response.json();
        if (response.ok) {
            document.getElementById("result").textContent = `Predicted Class: ${result.class}`;
        } else {
            document.getElementById("result").textContent = `Error: ${result.error}`;
        }
    } catch (error) {
        document.getElementById("result").textContent = `An error occurred: ${error.message}`;
    }
});