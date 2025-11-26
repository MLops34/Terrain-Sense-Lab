// Simple inline prediction animation
function animatePredictionResult(formattedPrice) {
    const resultElement = document.getElementById('prediction-result');
    const priceElement = document.getElementById('price-value');
    
    if (!resultElement || !priceElement) return;
    
    // Make sure the element is visible
    resultElement.classList.remove('d-none');
    
    // Scroll to result
    resultElement.scrollIntoView({ behavior: 'smooth' });
    
    // Get the numeric value
    let finalPrice = 0;
    try {
        finalPrice = parseFloat(formattedPrice.replace(/[^0-9.-]+/g, ''));
    } catch(e) {
        finalPrice = 0;
    }
    
    // Check if value is valid
    if (isNaN(finalPrice) || finalPrice <= 0) {
        priceElement.textContent = formattedPrice;
        return;
    }
    
    // Animate counting up
    let currentPrice = 0;
    const duration = 1500; // ms
    const interval = 30; // ms
    const steps = duration / interval;
    const increment = finalPrice / steps;
    
    const counter = setInterval(() => {
        currentPrice += increment;
        if (currentPrice >= finalPrice) {
            currentPrice = finalPrice;
            clearInterval(counter);
        }
        priceElement.textContent = '$' + currentPrice.toLocaleString('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        });
    }, interval);
    
    // Add highlight effect to the card
    resultElement.style.transition = "box-shadow 0.3s";
    resultElement.style.boxShadow = "0 0 20px rgba(94, 114, 228, 0.8)";
    
    setTimeout(() => {
        resultElement.style.boxShadow = "";
    }, 2000);
}

function updateProofVisualization(rawPrediction) {
    const proofCard = document.getElementById('prediction-proof');
    if (!proofCard || rawPrediction === undefined || rawPrediction === null) return;

    const min = parseFloat(proofCard.dataset.min);
    const max = parseFloat(proofCard.dataset.max);
    const median = parseFloat(proofCard.dataset.median);
    const marker = document.getElementById('proof-marker');
    const proofCurrent = document.getElementById('proof-current');
    const proofBadge = document.getElementById('proof-position');

    if (!isFinite(min) || !isFinite(max) || max === min) return;

    const ratio = Math.min(Math.max((rawPrediction - min) / (max - min), 0), 1);
    if (marker) {
        marker.style.left = `${ratio * 100}%`;
    }

    if (proofCurrent) {
        proofCurrent.textContent = rawPrediction.toLocaleString('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        });
    }

    if (proofBadge && isFinite(median)) {
        const delta = rawPrediction - median;
        const direction = delta >= 0 ? 'above' : 'below';
        const magnitude = Math.abs(delta).toLocaleString('en-US', {
            style: 'currency',
            currency: 'USD',
            maximumFractionDigits: 0
        });
        proofBadge.textContent = `${direction} median by ${magnitude}`;
        proofBadge.classList.toggle('bg-basil', delta >= 0);
        proofBadge.classList.toggle('bg-peach', delta < 0);
        proofBadge.classList.toggle('text-dark', true);
    }

    proofCard.classList.remove('d-none');
}

// Add to form submission event
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get form data
            const formData = new FormData(this);
            const submitBtn = this.querySelector('button[type="submit"]');
            
            // Update button
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Crafting...';
            submitBtn.disabled = true;
            
            // Send prediction request
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Reset button
                submitBtn.innerHTML = '<i class="fas fa-sparkles me-2"></i>Predict with Flair';
                submitBtn.disabled = false;
                
                // Show result
                if (data.success) {
                    animatePredictionResult(data.formatted_prediction);
                    updateProofVisualization(data.prediction);
                } else {
                    const resultElement = document.getElementById('prediction-result');
                    resultElement.classList.remove('d-none');
                    document.getElementById('price-value').textContent = 'Error: ' + data.error;
                    resultElement.scrollIntoView({ behavior: 'smooth' });
                }
            })
            .catch(error => {
                // Reset button
                submitBtn.innerHTML = '<i class="fas fa-sparkles me-2"></i>Predict with Flair';
                submitBtn.disabled = false;
                
                // Show error
                const resultElement = document.getElementById('prediction-result');
                resultElement.classList.remove('d-none');
                document.getElementById('price-value').textContent = 'Error: Something went wrong';
                resultElement.scrollIntoView({ behavior: 'smooth' });
                console.error('Error:', error);
            });
        });
    }
});