const vid = document.querySelector('#webcamVideo');
let intervalId = null;

var localstream;
let pX, pY, boundingWidth, boundingHeight;

function vidOff() {
    vid.pause();
    vid.src = "";
    vid.srcObject.getTracks()[0].stop();
}

function turnOffInferring() {
    // document.getElementById('infer').checked = false;
    chrome.storage.local.set({
        'infer': false
    }, () => {});
    chrome.extension.sendRequest({
        infer: false
    });
    vidOff();
    if (intervalId) {
        clearInterval(intervalId);
    }
}

function inferButtonClicked() {
    if (document.getElementById('infer').checked) {
        turnOnInferring()
    } else {
        turnOffInferring()
    }
}

function turnOnInferring() {
    setupCam();
    // document.getElementById('infer').checked = true;
    chrome.storage.local.set({
        'infer': true
    }, () => {});
    chrome.extension.sendRequest({
        infer: true
    });
    if (intervalId) {
        clearInterval(intervalId);
    }
    intervalId = setInterval(drawImage, 100);
}

function setupCam() {
    navigator.mediaDevices.getUserMedia({
        video: true
    }).then(mediaStream => {
        document.querySelector('#webcamVideo').srcObject = mediaStream;
        localstream = mediaStream;
    }).catch((error) => {
        console.warn(error);
    });
}


// document.querySelector("#capture").onclick = getImage;
// Setup checkbox with correct initial value.
chrome.storage.local.get('infer', items => {
    document.getElementById('infer').checked = !!items['infer'];
    document.getElementById('capture').innerHTML = !!items['infer']
});

const canvasTemp = document.createElement("canvas");

function drawImage(){

    let width = 227, height = 170.25;
    // document.getElementById("ratio").innerHTML = vid.videoWidth > 0 ? vid.videoHeight / vid.videoWidth : 0;

    canvasTemp.setAttribute('width', `${width}`); // clears the canvas
    canvasTemp.setAttribute('height', `${height}`); // clears the canvas
    canvasTemp.getContext('2d').drawImage(vid, 0, 0, width, height);

    var url = "http://localhost:8000/detector";
    var json = {imageSrc: canvasTemp.toDataURL()};
    json = JSON.stringify(json);

    // let rx, ry, rw, rh;
    fetch(url, {
        method: 'POST',
        body: json,
        headers: { 'Content-Type': 'application/json', "Access-Control-Allow-Origin": "*" },
        crossDomain: true
    }).then(res => res.json()).then(result => {

        var canvas = document.querySelector("#videoCanvas");
        
        canvas.setAttribute('width', `${width}`); // clears the canvas
        canvas.setAttribute('height', `${height}`); // clears the canvas
    
        var ctx = canvas.getContext('2d');
    
        ctx.drawImage(vid, 0, 0, canvas.width, canvas.height);

        let rx = parseFloat(result['rx'])
        let ry = parseFloat(result['ry'])
        let rw = parseFloat(result['rw'])
        let rh = parseFloat(result['rh'])
        
        pX = width * rx;
        pY = height * ry;
        boundingWidth = width * rw;
        boundingHeight = height * rh;
    
        ctx.rect(pX,pY,boundingWidth,boundingHeight);
        ctx.lineWidth = "3";
        ctx.strokeStyle = "green";    
        ctx.stroke();

        if (!document.getElementById('infer').checked) {
            canvas.setAttribute('width', `${width}`); // clears the canvas
            canvas.setAttribute('height', `${height}`); // clears the canvas
        }
    })
}

chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    document.getElementById("label_model").innerHTML = request.data;
  });

document.getElementById('infer').onclick = inferButtonClicked;