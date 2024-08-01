const express = require('express');
const { spawn } = require('child_process');
const WebSocket = require('ws');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3000;
let pythonProcess = null;
let simulateProcess = null;
let trainProcess = null;

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

app.get('/get-models', (req, res) => {
    const modelsDir = 'C:/Users/Enzo/Documents/Code_enzo/resnet18_test_/'; // Remplacez par le chemin réel de votre dossier de modèles
    fs.readdir(modelsDir, (err, files) => {
        if (err) {
            console.error(err);
            res.status(500).send('Erreur lors de la lecture du dossier de modèles');
        } else {
            const modelFiles = files.filter(file => file.endsWith('.pth'));
            res.json(modelFiles);
        }
    });
});

app.get('/start-python', (req, res) => {
    if (!pythonProcess) {
        const imageType = req.query.image_type || 'png';
        const folderPath = req.query.folder_path || '';
        const modelName = req.query.model_name || '';

        pythonProcess = spawn('C:/Users/Enzo/AppData/Local/Programs/Python/Python312/python.exe', 
            ['script.py', '-image_type', imageType, '-folder_path', folderPath, '-model_name', modelName]);

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
        const pythonInterpreter = 'C:/Users/Enzo/AppData/Local/Programs/Python/Python312/python.exe';
        
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


app.get('/train-model', (req, res) => {
    if (!trainProcess) {
        const modelName = req.query.model_name || '';

        trainProcess = spawn('C:/Users/Enzo/AppData/Local/Programs/Python/Python312/python.exe', 
            ['train.py', '-model_name', modelName]);

        trainProcess.stdout.on('data', (data) => {
            console.log(`stdout: ${data}`);
            wss.clients.forEach(client => {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(data.toString());
                }
            });
        });

        trainProcess.stderr.on('data', (data) => {
            console.error(`stderr: ${data}`);
        });

        trainProcess.on('close', (code) => {
            console.log(`Processus d'entraînement terminé avec le code ${code}`);
            trainProcess = null;
        });

        res.send('Entraînement du modèle lancé');
    } else {
        res.send('L\'entraînement est déjà en cours');
    }
});