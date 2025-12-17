document.addEventListener("DOMContentLoaded", () => {
    
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

    imageInput.value = null;

    let currentMaxK = 10; // Default max K

async function reloadDataset() {
    const dataset = datasetSelect.value;
    const kInput = document.getElementById('k-input');
    const kMsg = document.getElementById('k-limit-msg');
    
    // Lock UI
    processButton.disabled = true;
    processButton.textContent = "Loading...";
    
    try {
        const response = await fetch('http://localhost:5000/load_dataset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset: dataset })
        });
        const data = await response.json();

        if (response.ok) {
            // 1. SAVE THE LIMIT
            currentMaxK = data.max_k;
            
            // 2. UPDATE THE INPUT FIELD
            kInput.max = currentMaxK;
            kMsg.textContent = `Max allowed: ${currentMaxK} (Based on folders/images)`;
            
            // Reset value if it's currently too high
            if (parseInt(kInput.value) > currentMaxK) {
                kInput.value = currentMaxK;
            }

            toast.textContent = `Loaded ${dataset} (Max K=${currentMaxK})`;
            toast.classList.remove('hidden');
            setTimeout(() => toast.classList.add('hidden'), 3000);
        } else { 
            alert("Error: " + data.error); 
        }
    } catch (e) { 
        console.error(e); 
    } finally {
        processButton.disabled = false;
        processButton.textContent = "Run Clustering";
    }
}
    
    datasetSelect.addEventListener('change', reloadDataset);
    reloadDataset();

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
        formData.append('k', document.getElementById('k-input').value);

        try {
            const response = await fetch('http://localhost:5000/process_image', { 
                method: 'POST', 
                body: formData 
            });
            const data = await response.json();
            
            if (response.ok) {
                resultImage.src = "data:image/png;base64," + data.image_b64;
                
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

    runStatsButton.addEventListener('click', async () => { 
        // Get the user's chosen K
        const userK = parseInt(document.getElementById('k-input').value);
        
        // Validate
        if (userK > currentMaxK) {
            alert(`Please choose a K value less than or equal to ${currentMaxK}`);
            return;
        }
        if (userK > 20) {
            const confirmHigh = confirm(`Running stats up to K=${userK} might be slow. Continue?`);
            if (!confirmHigh) return;
        }

        statsLoader.classList.remove('hidden');
        statsSection.classList.add('hidden');
        runStatsButton.disabled = true;
        
        try {
            // SEND THE LIMIT TO BACKEND
            const response = await fetch('http://localhost:5000/run_statistics', { 
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ max_k: userK }) 
            });
            
            const data = await response.json(); 
            
            if (response.ok) {
                if(data.error) {
                    alert(data.error);
                } else {
                    displayScientificCharts(data);
                }
            }
        } catch (error) {
            console.error(error);
        } finally {
            statsLoader.classList.add('hidden');
            statsSection.classList.remove('hidden');
            runStatsButton.disabled = false;
        }
    });

    function displayScientificCharts(data) {
        const labels = data.var1.map(item => `K=${item.k}`);
        
        const inertiaV1 = data.var1.map(i => i.inertia);
        const inertiaV2 = data.var2.map(i => i.inertia);

        const silV1 = data.var1.map(i => i.silhouette);
        const silV2 = data.var2.map(i => i.silhouette);

        const timeV1 = data.var1.map(i => i.time);
        const timeV2 = data.var2.map(i => i.time);

        chartInertiaInstance = renderChart('chartInertia', chartInertiaInstance, labels, inertiaV1, inertiaV2, 'Inertia (Lower is Better)');
        chartSilInstance = renderChart('chartSilhouette', chartSilInstance, labels, silV1, silV2, 'Silhouette (Higher is Better)');
        chartTimeInstance = renderChart('chartTime', chartTimeInstance, labels, timeV1, timeV2, 'Execution Time (Seconds)');
    }

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
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.2)',
                        tension: 0.1
                    },
                    {
                        label: 'Var 2 (Random Points)',
                        data: data2,
                        borderColor: '#3b82f6',
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