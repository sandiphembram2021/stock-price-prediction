// Frontend Configuration
const CONFIG = {
    // API Configuration
    API_BASE_URL: 'http://localhost:8000/api/v1',
    
    // Chart Configuration
    CHART_COLORS: {
        primary: 'rgb(102, 126, 234)',
        secondary: 'rgb(118, 75, 162)',
        success: 'rgb(40, 167, 69)',
        danger: 'rgb(220, 53, 69)',
        warning: 'rgb(255, 193, 7)',
        info: 'rgb(23, 162, 184)'
    },
    
    // Default Values (matching roni.ipynb)
    DEFAULTS: {
        PREDICTION_DAYS: 30,
        TRAINING_EPOCHS: 50,
        BATCH_SIZE: 16,
        WINDOW_SIZE: 10  // Same as in roni.ipynb
    },
    
    // Popular Stock Symbols
    POPULAR_STOCKS: [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
        'META', 'NVDA', 'NFLX', 'BABA', 'V'
    ],
    
    // Refresh Intervals (in milliseconds)
    REFRESH_INTERVALS: {
        HEALTH_CHECK: 30000,  // 30 seconds
        AUTO_REFRESH: 300000  // 5 minutes
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
}
