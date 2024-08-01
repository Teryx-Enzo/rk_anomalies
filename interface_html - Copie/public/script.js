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

document.getElementById('startPythonButton').addEventListener('click', () => {
    const imageType = document.getElementById('imageType').value;
    const folderPath = document.getElementById('folderPath').value;
    const modelName = document.getElementById('modelSelect').value;

    fetch(`/start-python?image_type=${imageType}&folder_path=${encodeURIComponent(folderPath)}&model_name=${encodeURIComponent(modelName)}`)
        .then(response => response.text())
        .then(data => {
            console.log(data); // Log initial response
        })
        .catch(error => console.error('Erreur:', error));
});

document.getElementById('simulateButton').addEventListener('click', () => {
    const sourceDir = 'C:/Users/Enzo/Pictures/dataset/test_ds_30';
    const destDir = 'C:/Users/Enzo/Pictures/dataset';
    
    fetch(`/simulate?source_dir=${sourceDir}&dest_dir=${destDir}`)
        .then(response => response.text())
        .then(data => {
            console.log(data); // Log initial response
        })
        .catch(error => console.error('Erreur:', error));
});

const socket = new WebSocket('ws://localhost:3000');

socket.addEventListener('message', function (event) {
    const output = document.getElementById('pythonOutput');
    const statusIndicatorClass = document.getElementById('indicator_class');
    const statusIndicatorTime = document.getElementById('indicator_time');
    const timeOutput = document.getElementById('timeOutput');

    const message = event.data.split('\n').filter(line => line.trim() !== '');  
    if (message.length > 0) {
        if (message[0].startsWith('Temps')) {
            const times = message.map(line => parseFloat(line.split(': ')[1].replace(' ms', '')));
            updateChart(times);
        } else {
            const status = message[0].split(';')[0];
            const time = parseInt(message[0].split(';')[1]);

            if (status === 'G') {
                statusIndicatorClass.classList.remove('red');
                statusIndicatorClass.classList.add('green');
            } else {
                statusIndicatorClass.classList.remove('green');
                statusIndicatorClass.classList.add('red');
            }
    
            if (time < 300) {
                statusIndicatorTime.classList.remove('red');
                statusIndicatorTime.classList.add('green');
            } else {
                statusIndicatorTime.classList.remove('green');
                statusIndicatorTime.classList.add('red');
            }   
    
            timeOutput.innerText = `Temps: ${time} ms`;
            output.innerText += `${event.data}\n`;
        }   
    }
});

function updateChart(times) {
    const ctx = document.getElementById('executionChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: times.map((_, i) => `Itération ${i + 1}`),
            datasets: [{
                label: 'Temps d\'exécution (ms)',
                data: times,
                backgroundColor: times.map(time => time > 300 ? 'rgba(255, 99, 132, 0.2)' : 'rgba(75, 192, 192, 0.2)'),
                borderColor: times.map(time => time > 300 ? 'rgba(255, 99, 132, 1)' : 'rgba(75, 192, 192, 1)'),
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            },
            plugins: {
                annotation: {
                    annotations: {
                        line1: {
                            type: 'line',
                            yMin: 300,
                            yMax: 300,
                            borderColor: 'rgba(255, 0, 0, 0.5)',
                            borderWidth: 2,
                            label: {
                                enabled: true,
                                content: 'Seuil de 300 ms'
                            }
                        }
                    }
                }
            }
        }
    });
}
