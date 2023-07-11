const vid = document.querySelector('#webcamVideo');
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
}

function inferButtonClicked() {
    if (document.getElementById('infer').checked) {
        document.getElementById('capture').innerHTML = document.getElementById('infer').checked;
        turnOnInferring()
    } else {
        document.getElementById('capture').innerHTML = document.getElementById('infer').checked;
        turnOffInferring()
    }

}

function turnOnInferring() {

    // document.getElementById('infer').checked = true;
    chrome.storage.local.set({
        'infer': true
    }, () => {});
    chrome.extension.sendRequest({
        infer: true
    });
}
setupCam();

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

function drawImge(){
    var video = document.querySelector("#webcamVideo");

    let width = 227, height = 227 / video.videoWidth * video.videoHeight;

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
    
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // var str = result['content'];
        let rx = parseFloat(result['rx'])
        let ry = parseFloat(result['ry'])
        let rw = parseFloat(result['rw'])
        let rh = parseFloat(result['rh'])
        // document.getElementById("rx").innerHTML = rx;
        // document.getElementById("ry").innerHTML = ry;
        // document.getElementById("rw").innerHTML = rw;
        // document.getElementById("rh").innerHTML = rh;
        
        pX = width * rx;
        pY = height * ry;
        boundingWidth = width * rw;
        boundingHeight = height * rh;
    
        ctx.rect(pX,pY,boundingWidth,boundingHeight);
        ctx.lineWidth = "6";
        ctx.strokeStyle = "red";    
        ctx.stroke();
    })


    // var faceArea = 50;
    // var pX=canvas.width/2 - faceArea/2;
    // var pY=canvas.height/2 - faceArea/2;
    // let pX = width * rx;
    // let pY = height * ry;
    // let boundingWidth = width * rw;
    // let boundingHeight = height * rh;

    // ctx.rect(pX,pY,boundingWidth,boundingHeight);
    // ctx.lineWidth = "6";
    // ctx.strokeStyle = "red";    
    // ctx.stroke();


    // setTimeout(drawImge , 100);
}

// var video = document.querySelector("#webcamVideo");
// video.onplay = function() {
//     setTimeout(drawImge , 300);
// };
setInterval(drawImge, 100);

document.getElementById('infer').onclick = inferButtonClicked;