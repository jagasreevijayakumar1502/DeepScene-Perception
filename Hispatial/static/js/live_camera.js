const cameraElement = document.getElementById('camera');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const liveReport = document.getElementById('liveReport');
const captureCanvas = document.getElementById('captureCanvas');
const ctx = captureCanvas.getContext('2d');
let intervalId = null;
let stream = null;

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        cameraElement.srcObject = stream;
        cameraElement.play();
        intervalId = setInterval(captureFrame, 1800);
        liveReport.innerHTML = '<p>Live analysis started. Processing frames every 1.8 seconds.</p>';
    } catch (error) {
        liveReport.innerHTML = `<p style="color:#ff6b6b">Camera access failed: ${error.message}</p>`;
    }
}

function stopCamera() {
    if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
    }
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    liveReport.innerHTML = '<p>Live camera stopped.</p>';
}

async function captureFrame() {
    if (!cameraElement.videoWidth) {
        return;
    }
    captureCanvas.width = cameraElement.videoWidth;
    captureCanvas.height = cameraElement.videoHeight;
    ctx.drawImage(cameraElement, 0, 0, captureCanvas.width, captureCanvas.height);
    const imageData = captureCanvas.toDataURL('image/jpeg', 0.85);
    liveReport.innerHTML = '<p>Sending frame to HiSpatial engine...</p>';

    try {
        const response = await fetch('/process_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        const result = await response.json();
        if (response.ok) {
            renderLiveResult(result);
        } else {
            liveReport.innerHTML = `<p style="color:#ff6b6b">Error: ${result.error || 'Unknown error'}</p>`;
        }
    } catch (error) {
        liveReport.innerHTML = `<p style="color:#ff6b6b">Network error: ${error.message}</p>`;
    }
}

function renderLiveResult(result) {
    const sentences = result.narrative || [];
    const list = sentences.map(sentence => `<li>${sentence}</li>`).join('');
    liveReport.innerHTML = `
        <div>
            <p><strong>Detected objects:</strong> ${result.objects ? result.objects.length : 0}</p>
            <p><strong>Scene description:</strong></p>
            <ul>${list}</ul>
        </div>
    `;
}

startButton.addEventListener('click', startCamera);
stopButton.addEventListener('click', stopCamera);
window.addEventListener('beforeunload', stopCamera);
