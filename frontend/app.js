document.addEventListener("DOMContentLoaded", () => {
    
    // --- 1. Get Elements ---
    const uploadForm = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-file');
    const imagePreview = document.getElementById('image-preview');
    const algorithmSelect = document.getElementById('algorithm-select');
    const kSelect = document.getElementById('k-select');
    const processButton = document.getElementById('processButton');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('results-section');
    const resultsOutput = document.getElementById('results-output');
    const resultImage = document.getElementById('result-image');
    const centroidImage = document.getElementById('centroid-image');
    const datasetSelect = document.getElementById('dataset-select');
    const toast = document.getElementById('toast');
    
    // Stats Elements
    const runStatsButton = document.getElementById('run-stats-button');
    const clearStatsButton = document.getElementById('clear-stats-button');
    const statsLoader = document.getElementById('stats-loader');
    const statsSection = document.getElementById('statistics-section');
    const timeChartCanvas = document.getElementById('time-chart');
    
    let timeChart = null;

    // --- 2. Initialization ---
    imageInput.value = null;

    // --- 3. Dataset Reloading ---
// --- 3. Dataset Reloading ---
    async function reloadDataset() {
        const dataset = datasetSelect.value;
        console.log(`Reloading dataset: ${dataset}`);
        
        // LOCK THE BUTTONS
        processButton.disabled = true;
        processButton.textContent = "Loading Dataset...";
        datasetSelect.disabled = true; // Prevent spamming change
        
        try {
            const response = await fetch('http://localhost:5000/load_dataset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset: dataset })
            });
            if (response.ok) {
                toast.classList.remove('hidden');
                setTimeout(() => toast.classList.add('hidden'), 3000);
            } else { 
                alert("Failed to load dataset."); 
            }
        } catch (e) { 
            console.error(e); 
            alert("Error switching dataset."); 
        } finally {
            // UNLOCK THE BUTTONS
            processButton.disabled = false;
            processButton.textContent = "Run Clustering";
            datasetSelect.disabled = false;
        }
    }
    datasetSelect.addEventListener('change', reloadDataset);
    // Load default on startup
    reloadDataset();

    // --- 4. Main Process Submission ---
    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault(); 
        const file = imageInput.files[0];
        if (!file) {
            alert("Please select an image file first.");
            return;
        }
        
        setLoading(true);

        const formData = new FormData();
        formData.append('image', file);
        formData.append('algorithm', algorithmSelect.value);
        formData.append('k', kSelect.value);

        try {
            const response = await fetch('http://localhost:5000/process_image', { 
                method: 'POST', 
                body: formData 
            });
            const data = await response.json();
            
            if (response.ok) {
                resultImage.src = "data:image/png;base64," + data.image_b64;
                centroidImage.src = "data:image/png;base64," + data.ghost_b64;
                
                const outputText = `Method: ${data.algorithm}
Result: Assigned to ${data.person_label}
Internal Index: ${data.nearest_idx}`;
                
                resultsOutput.textContent = outputText;
                resultsSection.classList.remove('hidden');
            } else {
                alert(data.error || 'Unknown error');
            }
        } catch (error) {
            console.error('Error:', error);
            resultsOutput.textContent = `Error connecting to backend:\n${error.message}`;
        } finally {
            setLoading(false);
        }
    });

    // --- 5. Image Preview ---
    imageInput.addEventListener('change', async () => {
        const file = imageInput.files[0];
        if (!file) { imagePreview.style.display = 'none'; return; }
        
        const formData = new FormData();
        formData.append('image', file);
        try {
            const response = await fetch('http://localhost:5000/preview', { method: 'POST', body: formData });
            const data = await response.json();
            if (data.image_b64) {
                imagePreview.src = "data:image/png;base64," + data.image_b64;
                imagePreview.style.display = 'block';
            }
        } catch (e) { console.error(e); }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            processButton.disabled = true;
            processButton.textContent = 'Processing...';
            loader.classList.remove('hidden');
            resultsSection.classList.add('hidden'); 
        } else {
            processButton.disabled = false;
            processButton.textContent = 'Run Clustering';
            loader.classList.add('hidden');
        }
    }

    // --- 6. Statistics Logic ---
    runStatsButton.addEventListener('click', async () => { 
        statsLoader.classList.remove('hidden');
        statsSection.classList.add('hidden');
        
        try {
            const response = await fetch('http://localhost:5000/run_statistics', { method: 'POST' });
            const data = await response.json(); 
            if (response.ok) {
                displayCharts(data);
            }
        } catch (error) {
            console.error(error);
        } finally {
            statsLoader.classList.add('hidden');
            statsSection.classList.remove('hidden');
        }
    });

    function displayCharts(data) {
        if (timeChart) timeChart.destroy();
        
        const combined = [...data.classification, ...data.preprocessing];
        const labels = combined.map(d => d.name);
        const times = combined.map(d => d.time_ms);
        
        timeChart = new Chart(timeChartCanvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Execution Time (ms)',
                    data: times,
                    backgroundColor: 'rgba(255, 99, 132, 0.6)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: { beginAtZero: true, grid: { color: '#374151' } },
                    x: { grid: { color: '#374151' } }
                },
                plugins: { legend: { labels: { color: '#d1d5db' } } }
            }
        });
    }

    clearStatsButton.addEventListener('click', () => {
        if (timeChart) timeChart.destroy();
        statsSection.classList.add('hidden');
    });
});