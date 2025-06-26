// Stock Price Prediction Frontend Application
class StockPredictionApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000/api/v1';
        this.chart = null;
        this.currentData = null;
        
        this.initializeApp();
        this.setupEventListeners();
        this.checkApiHealth();
    }

    // Initialize the application
    initializeApp() {
        this.setDefaultDates();
        this.showToast('Application initialized', 'info');
    }

    // Set default training dates
    setDefaultDates() {
        const endDate = new Date();
        const startDate = new Date();
        startDate.setFullYear(endDate.getFullYear() - 1);
        
        document.getElementById('trainEndDate').value = endDate.toISOString().split('T')[0];
        document.getElementById('trainStartDate').value = startDate.toISOString().split('T')[0];
    }

    // Setup event listeners
    setupEventListeners() {
        document.getElementById('predictBtn').addEventListener('click', () => this.makePrediction());
        document.getElementById('trainBtn').addEventListener('click', () => this.toggleTrainingPanel());
        document.getElementById('refreshBtn').addEventListener('click', () => this.refreshData());
        document.getElementById('startTrainingBtn').addEventListener('click', () => this.startTraining());
        document.getElementById('cancelTrainingBtn').addEventListener('click', () => this.hideTrainingPanel());
        
        // Enter key support for stock symbol input
        document.getElementById('stockSymbol').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.makePrediction();
            }
        });
    }

    // API Health Check
    async checkApiHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const data = await response.json();
            
            this.updateStatus('apiStatus', data.status, data.status === 'healthy' ? 'healthy' : 'unhealthy');
            this.updateStatus('modelStatus', data.model_status, data.model_status === 'loaded' ? 'healthy' : 'unhealthy');
            this.updateLastUpdated();
            
        } catch (error) {
            this.updateStatus('apiStatus', 'Error', 'unhealthy');
            this.updateStatus('modelStatus', 'Unknown', 'unhealthy');
            this.showToast('Failed to connect to API', 'error');
        }
    }

    // Update status display
    updateStatus(elementId, text, status) {
        const element = document.getElementById(elementId);
        element.textContent = text;
        element.className = `status-value status-${status}`;
    }

    // Update last updated timestamp
    updateLastUpdated() {
        document.getElementById('lastUpdated').textContent = new Date().toLocaleTimeString();
    }

    // Make stock price prediction
    async makePrediction() {
        const symbol = document.getElementById('stockSymbol').value.trim().toUpperCase();
        const days = parseInt(document.getElementById('predictionDays').value);

        if (!symbol) {
            this.showToast('Please enter a stock symbol', 'error');
            return;
        }

        this.showLoading(true);
        
        try {
            let endpoint, requestData;
            
            if (days === 1) {
                endpoint = `${this.apiBaseUrl}/predict/single`;
                requestData = { symbol, days };
            } else {
                endpoint = `${this.apiBaseUrl}/predict/future`;
                requestData = { symbol, days };
            }

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            if (days === 1) {
                this.displaySinglePrediction(data);
            } else {
                this.displayFuturePredictions(data);
            }
            
            this.updateChart(data, days);
            this.loadPredictionHistory(symbol);
            this.showToast('Prediction completed successfully', 'success');
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.showToast(`Prediction failed: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    // Display single day prediction
    displaySinglePrediction(data) {
        document.getElementById('currentPrice').textContent = 
            data.current_price ? `$${data.current_price.toFixed(2)}` : '--';
        document.getElementById('predictedPrice').textContent = `$${data.predicted_price.toFixed(2)}`;
        
        if (data.current_price) {
            const change = data.predicted_price - data.current_price;
            const changePercent = (change / data.current_price) * 100;
            
            document.getElementById('priceChange').textContent = `$${change.toFixed(2)}`;
            document.getElementById('priceChangePercent').textContent = `${changePercent.toFixed(2)}%`;
            
            // Apply color coding
            const changeClass = change >= 0 ? 'positive' : 'negative';
            document.getElementById('priceChange').className = `summary-value ${changeClass}`;
            document.getElementById('priceChangePercent').className = `summary-value ${changeClass}`;
        }
    }

    // Display future predictions
    displayFuturePredictions(data) {
        if (data.predictions && data.predictions.length > 0) {
            const lastPrediction = data.predictions[data.predictions.length - 1];
            document.getElementById('predictedPrice').textContent = `$${lastPrediction.price.toFixed(2)}`;
            document.getElementById('currentPrice').textContent = '--';
            document.getElementById('priceChange').textContent = '--';
            document.getElementById('priceChangePercent').textContent = '--';
        }
    }

    // Update chart with prediction data
    updateChart(data, days) {
        const ctx = document.getElementById('priceChart').getContext('2d');
        
        if (this.chart) {
            this.chart.destroy();
        }

        let chartData, chartLabels;
        
        if (days === 1) {
            // Single prediction chart
            chartLabels = ['Current', 'Predicted'];
            chartData = [
                data.current_price || 0,
                data.predicted_price
            ];
        } else {
            // Future predictions chart
            chartLabels = data.predictions.map(p => {
                const date = new Date(p.date);
                return date.toLocaleDateString();
            });
            chartData = data.predictions.map(p => p.price);
        }

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartLabels,
                datasets: [{
                    label: 'Stock Price',
                    data: chartData,
                    borderColor: 'rgb(102, 126, 234)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: 'rgb(102, 126, 234)',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `${document.getElementById('stockSymbol').value.toUpperCase()} Price Prediction`,
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            }
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    }
                },
                elements: {
                    point: {
                        hoverRadius: 8
                    }
                }
            }
        });
    }

    // Load prediction history
    async loadPredictionHistory(symbol) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/predictions/${symbol}?limit=10`);
            if (response.ok) {
                const data = await response.json();
                this.displayPredictionHistory(data.predictions);
            }
        } catch (error) {
            console.error('Failed to load prediction history:', error);
        }
    }

    // Display prediction history in table
    displayPredictionHistory(predictions) {
        const tbody = document.getElementById('predictionsTableBody');
        
        if (!predictions || predictions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="no-data">No predictions available</td></tr>';
            return;
        }

        tbody.innerHTML = predictions.map(pred => {
            const predDate = new Date(pred.prediction_date).toLocaleDateString();
            const targetDate = new Date(pred.target_date).toLocaleDateString();
            const predictedPrice = `$${pred.predicted_price.toFixed(2)}`;
            const actualPrice = pred.actual_price ? `$${pred.actual_price.toFixed(2)}` : '--';
            
            let accuracy = '--';
            if (pred.actual_price) {
                const error = Math.abs(pred.predicted_price - pred.actual_price);
                const accuracyPercent = ((1 - error / pred.actual_price) * 100).toFixed(1);
                accuracy = `${accuracyPercent}%`;
            }

            return `
                <tr>
                    <td>${predDate}</td>
                    <td>${targetDate}</td>
                    <td>${predictedPrice}</td>
                    <td>${actualPrice}</td>
                    <td>${accuracy}</td>
                </tr>
            `;
        }).join('');
    }

    // Toggle training panel
    toggleTrainingPanel() {
        const panel = document.getElementById('trainingPanel');
        panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    }

    // Hide training panel
    hideTrainingPanel() {
        document.getElementById('trainingPanel').style.display = 'none';
    }

    // Start model training
    async startTraining() {
        const symbol = document.getElementById('stockSymbol').value.trim().toUpperCase();
        const startDate = document.getElementById('trainStartDate').value;
        const endDate = document.getElementById('trainEndDate').value;
        const epochs = parseInt(document.getElementById('epochs').value);
        const batchSize = parseInt(document.getElementById('batchSize').value);

        if (!symbol || !startDate || !endDate) {
            this.showToast('Please fill in all training parameters', 'error');
            return;
        }

        this.showTrainingProgress(true);
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/train`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    symbol,
                    start_date: startDate,
                    end_date: endDate,
                    epochs,
                    batch_size: batchSize
                })
            });

            const data = await response.json();
            
            if (data.success) {
                this.showToast('Training started successfully', 'success');
                this.simulateTrainingProgress();
            } else {
                this.showToast(`Training failed: ${data.message}`, 'error');
                this.showTrainingProgress(false);
            }
            
        } catch (error) {
            console.error('Training error:', error);
            this.showToast(`Training failed: ${error.message}`, 'error');
            this.showTrainingProgress(false);
        }
    }

    // Show/hide training progress
    showTrainingProgress(show) {
        const progress = document.getElementById('trainingProgress');
        progress.style.display = show ? 'block' : 'none';
        
        if (!show) {
            document.getElementById('progressFill').style.width = '0%';
            document.getElementById('trainingStatus').textContent = 'Training in progress...';
        }
    }

    // Simulate training progress (since training runs in background)
    simulateTrainingProgress() {
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress >= 100) {
                progress = 100;
                clearInterval(interval);
                document.getElementById('trainingStatus').textContent = 'Training completed!';
                setTimeout(() => {
                    this.showTrainingProgress(false);
                    this.hideTrainingPanel();
                    this.checkApiHealth();
                }, 2000);
            }
            
            document.getElementById('progressFill').style.width = `${progress}%`;
            document.getElementById('trainingStatus').textContent = 
                `Training in progress... ${Math.round(progress)}%`;
        }, 1000);
    }

    // Refresh data
    async refreshData() {
        this.updateLastUpdated();
        await this.checkApiHealth();
        
        const symbol = document.getElementById('stockSymbol').value.trim();
        if (symbol) {
            await this.loadPredictionHistory(symbol);
        }
        
        this.showToast('Data refreshed', 'info');
    }

    // Show/hide loading overlay
    showLoading(show) {
        document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
    }

    // Show toast notification
    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 5000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new StockPredictionApp();
});
