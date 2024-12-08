function handleImagePreview() {
    const fileInput = document.getElementById('imageInput');
    const previewContainer = document.getElementById('imagePreviewContainer');
    const uploadSection = document.getElementById('uploadSection');
    const changeImageButton = document.getElementById('changeImageButton');

    fileInput.addEventListener('change', function () {
        const file = fileInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                previewContainer.innerHTML = `<img src="${e.target.result}" alt="MRI Image Preview">`;
                uploadSection.classList.add('hidden'); // Hide the upload section
                changeImageButton.classList.remove('hidden'); // Show the Change Image button
            };
            reader.readAsDataURL(file);
        }
    });

    // Change image functionality
    changeImageButton.addEventListener('click', function () {
        fileInput.click(); // Trigger the file input click to change the image
    });
}

document.getElementById("uploadForm").addEventListener("click", async (event) => {
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
            const ans=document.createElement('p');
            ans.textContent = `Predicted Class: ${result.class}`;
            document.getElementById("result").innerHTML="";
            document.getElementById("result").appendChild(ans);
        } else {
            document.getElementById("result").textContent = `Error: ${result.error}`;
        }
    } catch (error) {
        document.getElementById("result").textContent = `An error occurred: ${error.message}`;
    }
});

handleImagePreview();