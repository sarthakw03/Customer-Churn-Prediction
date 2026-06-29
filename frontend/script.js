class ChurnPredictorApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:5000';
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkBackendHealth();
    }

    bindEvents() {
        document.getElementById('churnForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.predictChurn();
        });

        document.getElementById('loadSample').addEventListener('click', () => {
            this.loadSampleData();
        });
    }

    async checkBackendHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const data = await response.json();
            
            if (data.status === 'healthy') {
                console.log('Backend is running');
            } else {
                this.showError('Backend is not responding properly');
            }
        } catch (error) {
            this.showError('Cannot connect to backend server. Make sure the Flask server is running on port 5000.');
        }
    }

    async loadSampleData() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/sample`);
            const sampleData = await response.json();
            
            // Fill form with sample data
            Object.keys(sampleData).forEach(key => {
                const element = document.getElementById(key);
                if (element) {
                    element.value = sampleData[key];
                }
            });
            
            this.showSuccess('Sample data loaded successfully!');
        } catch (error) {
            this.showError('Failed to load sample data');
        }
    }

    async predictChurn() {
        const formData = this.getFormData();
        
        if (!this.validateForm(formData)) {
            return;
        }

        this.showLoading(true);

        try {
            const response = await fetch(`${this.apiBaseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Prediction failed');
            }

            this.displayResults(result);
            this.showLoading(false);

        } catch (error) {
            this.showLoading(false);
            this.showError(`Prediction error: ${error.message}`);
        }
    }

    getFormData() {
        const form = document.getElementById('churnForm');
        const formData = new FormData(form);
        const data = {};

        for (let [key, value] of formData.entries()) {
            // Convert numeric fields to numbers
            if (['age', 'tenure', 'monthly_charges', 'total_charges', 
                 'monthly_usage_gb', 'support_calls', 'payment_delay'].includes(key)) {
                data[key] = parseFloat(value);
            } else {
                data[key] = value;
            }
        }

        return data;
    }

    validateForm(data) {
        const required = ['age', 'tenure', 'monthly_charges', 'contract_type'];
        
        for (let field of required) {
            if (!data[field] || data[field] === '') {
                this.showError(`Please fill in the ${field} field`);
                return false;
            }
        }

        if (data.age < 18 || data.age > 100) {
            this.showError('Please enter a valid age (18-100)');
            return false;
        }

        return true;
    }

    displayResults(result) {
        const resultsSection = document.getElementById('resultsSection');
        const riskLevel = document.getElementById('riskLevel');
        const probability = document.getElementById('probability');
        const recommendations = document.getElementById('recommendations');

        // Update risk level
        riskLevel.className = 'risk-level';
        riskLevel.classList.add(result.will_churn ? 'risk-high' : 'risk-low');
        riskLevel.textContent = `Churn Risk: ${result.prediction}`;

        // Update probability
        probability.textContent = `Churn Probability: ${result.churn_probability}% (${result.confidence} confidence)`;

        // Update recommendations
        recommendations.innerHTML = `
            <h3>Recommendations:</h3>
            <ul>
                ${result.recommendation.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        `;

        // Show results section
        resultsSection.style.display = 'block';
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    showLoading(show) {
        const loading = document.getElementById('loading');
        loading.style.display = show ? 'flex' : 'none';
    }

    showError(message) {
        alert(`Error: ${message}`);
    }

    showSuccess(message) {
        alert(`Success: ${message}`);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ChurnPredictorApp();
});