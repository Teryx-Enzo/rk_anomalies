
const socket = new WebSocket('ws://localhost:3000');

document.getElementById('compareButton').addEventListener('click', () => {
    const fileInput1 = document.getElementById('csvFile1');
    const fileInput2 = document.getElementById('csvFile2');
    const output = document.getElementById('compareOutput');
    output.innerText = '';

    if (fileInput1.files.length === 0 || fileInput2.files.length === 0) {
        alert('Veuillez sélectionner deux fichiers CSV.');
        return;
    }

    const formData = new FormData();
    formData.append('file1', fileInput1.files[0]);
    formData.append('file2', fileInput2.files[0]);

    fetch('/compare-csv', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(data => {
        output.innerText = data;
    })
    .catch(error => console.error('Erreur:', error));
});
