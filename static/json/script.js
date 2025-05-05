// Access the video element and set up the webcam stream
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const output = document.getElementById("output").querySelector("span");

let isDetecting = false;  // Flag to control real-time detection
let intervalId = null;    // Stores the setInterval ID

// Start the webcam stream
navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {
        video.srcObject = stream;
    })
    .catch(function (error) {
        console.error("Error accessing the camera: ", error);
    });

// Function to start real-time detection
function startDetection() {
    if (isDetecting) return;  // Avoid multiple intervals
    isDetecting = true;

    intervalId = setInterval(() => {
        captureAndSendImage();
    }, 1000);  // Capture image every second
}

// Function to stop real-time detection
function stopDetection() {
    isDetecting = false;
    clearInterval(intervalId);
}

// Capture image from video stream and send to Flask backend
function captureAndSendImage() {
    // Ensure canvas size matches video dimensions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw the current frame from the video onto the canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert the canvas image to a base64 string (JPEG format)
    const imageBase64 = canvas.toDataURL('image/jpeg').split(',')[1];

    // Send image to Flask backend for prediction
    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageBase64 })
    })
    .then(response => response.json())
    .then(data => {
        // Display the detected sign
        output.innerText = data.prediction || "No sign detected!";
    })
    .catch(error => {
        console.error('Error:', error);
        output.innerText = "Error detecting sign.";
    });
}
