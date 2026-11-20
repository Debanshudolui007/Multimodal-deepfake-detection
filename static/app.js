const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const resetBtn = document.getElementById('reset-btn');

// Drag and drop event listeners
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

dropZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

resetBtn.addEventListener('click', () => {
    results.classList.add('hidden');
    dropZone.classList.remove('hidden');
    fileInput.value = '';
});

function handleFile(file) {
    if (!file.type.match('video/mp4')) {
        alert('Please upload an MP4 video.');
        return;
    }

    dropZone.classList.add('hidden');
    loading.classList.remove('hidden');

    const formData = new FormData();
    formData.append('video', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loading.classList.add('hidden');
        if (data.error) {
            alert("Error: " + data.error);
            dropZone.classList.remove('hidden');
            return;
        }
        showResults(data);
    })
    .catch(error => {
        loading.classList.add('hidden');
        alert("An error occurred during prediction.");
        dropZone.classList.remove('hidden');
        console.error(error);
    });
}

function showResults(data) {
    results.classList.remove('hidden');
    
    const predText = document.getElementById('prediction-text');
    const confFill = document.getElementById('confidence-fill');
    const confText = document.getElementById('confidence-text');
    const chart = document.getElementById('attention-chart');
    
    predText.textContent = data.prediction;
    
    if (data.prediction === "FAKE") {
        predText.style.color = 'var(--fake)';
        confFill.style.background = 'var(--fake)';
    } else {
        predText.style.color = 'var(--real)';
        confFill.style.background = 'var(--real)';
    }
    
    const confPct = (data.confidence * 100).toFixed(1);
    
    // Animate width
    setTimeout(() => {
        confFill.style.width = `${confPct}%`;
    }, 100);
    
    confText.textContent = `Confidence: ${confPct}%`;
    chart.src = data.attention_chart;
}
