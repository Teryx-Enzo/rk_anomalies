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
    const dataPath = document.getElementById('dataPath').value;
    const output = document.getElementById('trainOutput');
    output.innerText = '';

    fetch(`/train-model?model_name=${encodeURIComponent(modelName)}&data_path=${encodeURIComponent(dataPath)}`)
        .then(response => response.text())
        .then(data => {
            console.log(data); // Log initial response
        })
        .catch(error => console.error('Erreur:', error));
});

// Configuration de WebSocket pour écouter les messages du serveur
const socket = new WebSocket('ws://localhost:3000');

socket.addEventListener('message', function (event) {
    const output = document.getElementById('trainOutput');
    output.innerText += `${event.data}\n`;
    output.scrollTop = output.scrollHeight; // Scroll to the bottom
});
