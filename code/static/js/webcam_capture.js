// Manual webcam preview and capture logic for registration and capture.html
let video, canvas, context;

function startWebcamPreview() {
    video = document.getElementById('webcamVideo');
    canvas = document.getElementById('webcamCanvas');
    context = canvas.getContext('2d');
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            video.srcObject = stream;
            video.play();
        })
        .catch(function(err) {
            alert('Could not access webcam: ' + err);
        });
}

function captureImage() {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    // Stop webcam
    let stream = video.srcObject;
    let tracks = stream.getTracks();
    tracks.forEach(track => track.stop());
    // Show captured image
    document.getElementById('capturedImage').src = canvas.toDataURL('image/jpeg');
    document.getElementById('capturedImage').style.display = 'block';
    document.getElementById('retakeBtn').style.display = 'inline-block';
    document.getElementById('submitBtn').style.display = 'inline-block';
    document.getElementById('manualCapture').style.display = 'none';
}

function retakeImage() {
    document.getElementById('capturedImage').style.display = 'none';
    document.getElementById('retakeBtn').style.display = 'none';
    document.getElementById('submitBtn').style.display = 'none';
    document.getElementById('manualCapture').style.display = 'inline-block';
    startWebcamPreview();
}