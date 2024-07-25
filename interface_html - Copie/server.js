const express = require('express');
const { spawn } = require('child_process');
const WebSocket = require('ws');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3000;
let pythonProcess = null;
let simulateProcess = null;

app.use(express.static('public'));

const server = app.listen(PORT, () => {
    console.log(`Serveur en cours d'exécution sur http://localhost:${PORT}`);
});

const wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
    console.log('Client connecté');

    ws.on('close', () => {
        console.log('Client déconnecté');
    });
});

app.get('/start-python', (req, res) => {
    if (!pythonProcess) {
        const imageType = req.query.image_type || 'png'; // Par défaut à 'png' si aucun type n'est spécifié
        // Spécifiez le chemin vers l'interpréteur Python de votre environnement virtuel
        pythonProcess = spawn('C:/Users/Enzo/AppData/Local/Programs/Python/Python312/python.exe', ['script.py' ,'-image_type',imageType]);

        pythonProcess.stdout.on('data', (data) => {
            console.log(`stdout: ${data}`);
            wss.clients.forEach(client => {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(data.toString());
                }
            });
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error(`stderr: ${data}`);
        });

        pythonProcess.on('close', (code) => {
            console.log(`Processus Python terminé avec le code ${code}`);
            pythonProcess = null;
        });

        res.send('Script Python lancé');
    } else {
        res.send('Le script Python est déjà en cours d\'exécution');
    }
});



app.get('/simulate', (req, res) => {
    if (!simulateProcess) {
        const sourceDir = req.query.source_dir || 'source_images';
        const destDir = req.query.dest_dir || 'images';
        const pythonInterpreter = 'C:/Users/Enzo/AppData/Local/Programs/Python/Python312/python.exe'; // Sur Windows : 'venv\\Scripts\\python.exe'
        console.log(`commande : ${sourceDir}`)
        simulateProcess = spawn(pythonInterpreter, ['simulate.py', '-source_dir', sourceDir, '-dest_dir', destDir]);

        let output = '';

        simulateProcess.stdout.on('data', (data) => {
            
            output += data.toString();
        });

        simulateProcess.stderr.on('data', (data) => {
            console.error(`stderr: ${data}`);
        });

        simulateProcess.on('close', (code) => {
            console.log(`Simulation terminée avec le code ${code}`);
            simulateProcess = null;
            // Envoyer les temps d'exécution au client
            wss.clients.forEach(client => {
                if (client.readyState === WebSocket.OPEN) {
                    console.log(`stdout: ${output}`);
                    client.send(output);
                }
            });
            res.send('Simulation terminée');
        });

    } else {
        res.send('La simulation est déjà en cours');
    }
});
