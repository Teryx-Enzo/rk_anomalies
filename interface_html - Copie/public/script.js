
document.getElementById('startPythonButton').addEventListener('click', () => {
    const imageType = document.getElementById('imageType').value;
    fetch(`/start-python?image_type=${imageType}`)
        .then(response => response.text())
        .then(data => {
            console.log(data); // Log initial response
        })
        .catch(error => console.error('Erreur:', error));
});



document.getElementById('simulateButton').addEventListener('click', () => {
    const sourceDir = 'C:/Users/Enzo/Pictures/dataset/test_ds_30';
    const destDir = 'C:/Users/Enzo/Pictures/dataset';
    
    //  const destDir = 'C:/Users/Enzo/Pictures/dataset' // Spécifiez le dossier source ici
    //const sourceDir = 'C:/Users/Enzo/Pictures/Images_triees_2/nouveaux_defauts'; // Spécifiez le dossier de destination ici
    
        

    fetch(`/simulate?source_dir=${sourceDir}&dest_dir=${destDir}`)
        .then(response => response.text())
        .then(data => {
            console.log(data); // Log initial response
        })
        .catch(error => console.error('Erreur:', error));
    
});

// Configuration de WebSocket pour écouter les messages du serveur
const socket = new WebSocket('ws://localhost:3000');

socket.addEventListener('message', function (event) {
    const output = document.getElementById('pythonOutput');
    const statusIndicatorClass = document.getElementById('indicator_class');
    const statusIndicatorTime = document.getElementById('indicator_time');
    const timeOutput = document.getElementById('timeOutput');

    // Parse the message from the server
    const message = event.data.split('\n').filter(line => line.trim() !== '');  
    if (message.length > 0) {
        if (message[0].startsWith('Temps')) {
            // Extraire les temps d'exécution pour le graphique
            const times = message.map(line => parseFloat(line.split(': ')[1].replace(' ms', '')));
            updateChart(times);
        } else {
            // Mise à jour de l'indicateur de statut et du temps
            const status = message[0].split(';')[0]; // G or NG
            const time = parseInt(message[0].split(';')[1]); // Time in ms
    // Update the status indicator
    if (status === 'G') {
        statusIndicatorClass.classList.remove('red');
        statusIndicatorClass.classList.add('green');
    } else {
        statusIndicatorClass.classList.remove('green');
        statusIndicatorClass.classList.add('red');
    }
    
    // Update the status indicator
    if (time < 300) {
        statusIndicatorTime.classList.remove('red');
        statusIndicatorTime.classList.add('green');
    } else {
        statusIndicatorTime.classList.remove('green');
        statusIndicatorTime.classList.add('red');
    }   

    // Update the time output
    timeOutput.innerText = `Temps: ${time} ms`;

    // Update the output log
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