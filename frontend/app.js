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
    
    // Chart Instances
    let chartInertiaInstance = null;
    let chartSilInstance = null;
    let chartTimeInstance = null;

    // --- 2. Initialization ---
    imageInput.value = null;

    // --- 3. Dataset Reloading ---
    async function reloadDataset() {
        const dataset = datasetSelect.value;
        console.log(`Reloading dataset: ${dataset}`);
        
        // LOCK THE UI
        processButton.disabled = true;
        processButton.textContent = "Loading Dataset...";
        datasetSelect.disabled = true;
        runStatsButton.disabled = true;
        
        try {
            const response = await fetch('http://localhost:5000/load_dataset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset: dataset })
            });
            const data = await response.json();

            if (response.ok) {
                // Show Toast
                toast.textContent = `Loaded ${dataset} (${data.shape[0]}x${data.shape[1]})`;
                toast.classList.remove('hidden');
                setTimeout(() => toast.classList.add('hidden'), 3000);
            } else { 
                alert("Failed to load dataset: " + data.error); 
            }
        } catch (e) { 
            console.error(e); 
            alert("Error connecting to backend."); 
        } finally {
            // UNLOCK THE UI
            processButton.disabled = false;
            processButton.textContent = "Run Clustering";
            datasetSelect.disabled = false;
            runStatsButton.disabled = false;
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
                
                // FIX: Check for 'ghost_b64' or 'centroid_b64' just in case
                if(data.ghost_b64) {
                    centroidImage.src = "data:image/png;base64," + data.ghost_b64;
                } else if (data.centroid_b64) {
                    centroidImage.src = "data:image/png;base64," + data.centroid_b64;
                }
                
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

    // --- 6. Statistics Logic (UPDATED FOR 3 CHARTS) ---
    runStatsButton.addEventListener('click', async () => { 
        statsLoader.classList.remove('hidden');
        statsSection.classList.add('hidden');
        runStatsButton.disabled = true;
        
        try {
            const response = await fetch('http://localhost:5000/run_statistics', { method: 'POST' });
            const data = await response.json(); 
            
            if (response.ok) {
                if(data.error) {
                    alert(data.error);
                } else {
                    displayScientificCharts(data);
                }
            } else {
                alert("Server error running statistics");
            }
        } catch (error) {
            console.error(error);
            alert("Failed to connect to backend for statistics.");
        } finally {
            statsLoader.classList.add('hidden');
            statsSection.classList.remove('hidden');
            runStatsButton.disabled = false;
        }
    });

    function displayScientificCharts(data) {
        // Prepare Data
        const labels = data.var1.map(item => `K=${item.k}`);
        
        // Extract Metrics
        const inertiaV1 = data.var1.map(i => i.inertia);
        const inertiaV2 = data.var2.map(i => i.inertia);

        const silV1 = data.var1.map(i => i.silhouette);
        const silV2 = data.var2.map(i => i.silhouette);

        const timeV1 = data.var1.map(i => i.time);
        const timeV2 = data.var2.map(i => i.time);

        // Render 3 Charts
        chartInertiaInstance = renderChart('chartInertia', chartInertiaInstance, labels, inertiaV1, inertiaV2, 'Inertia (Lower is Better)');
        chartSilInstance = renderChart('chartSilhouette', chartSilInstance, labels, silV1, silV2, 'Silhouette (Higher is Better)');
        chartTimeInstance = renderChart('chartTime', chartTimeInstance, labels, timeV1, timeV2, 'Execution Time (Seconds)');
    }

    // Generic Chart Renderer
    function renderChart(canvasId, chartInstance, labels, data1, data2, title) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        
        if (chartInstance) chartInstance.destroy();

        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Var 1 (Random Clusters)',
                        data: data1,
                        borderColor: '#ef4444', // Red
                        backgroundColor: 'rgba(239, 68, 68, 0.2)',
                        tension: 0.1
                    },
                    {
                        label: 'Var 2 (Random Points)',
                        data: data2,
                        borderColor: '#3b82f6', // Blue
                        backgroundColor: 'rgba(59, 130, 246, 0.2)',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: { display: false, text: title },
                    legend: { labels: { color: '#d1d5db' } },
                    tooltip: { mode: 'index', intersect: false }
                },
                scales: {
                    y: { 
                        beginAtZero: false, 
                        grid: { color: '#374151' },
                        ticks: { color: '#9ca3af' }
                    },
                    x: { 
                        grid: { color: '#374151' },
                        ticks: { color: '#9ca3af' }
                    }
                },
                interaction: { mode: 'nearest', axis: 'x', intersect: false }
            }
        });
    }

    clearStatsButton.addEventListener('click', () => {
        if (chartInertiaInstance) chartInertiaInstance.destroy();
        if (chartSilInstance) chartSilInstance.destroy();
        if (chartTimeInstance) chartTimeInstance.destroy();
        statsSection.classList.add('hidden');
    });
});