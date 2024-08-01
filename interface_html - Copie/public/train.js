document.addEventListener('DOMContentLoaded', () => {
    fetch('/get-models')
        .then(response => response.json())
        .then(models => {
            const modelSelect = document.getElementById('modelSelect');
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.text = model;
                modelSelect.appendChild(option);
            });
        })
        .catch(error => console.error('Erreur:', error));
});

document.getElementById('trainModelButton').addEventListener('click', () => {
    const modelName = document.getElementById('modelSelect').value;

    fetch(`/train-model?model_name=${encodeURIComponent(modelName)}`)
        .then(response => response.text())
        .then(data => {
            const output = document.getElementById('trainOutput');
            output.innerText = data;
        })
        .catch(error => console.error('Erreur:', error));
});
