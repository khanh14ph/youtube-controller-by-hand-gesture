const vid = document.querySelector('#webcamVideo');
let label = "no_gesture"
// const image = document.querySelector("#capturedimage");

var intervalId = null;
// Do first-time setup to gain access to webcam, if necessary.
chrome.runtime.onInstalled.addListener((details) => {
    if (details.reason.search(/install/g) === -1) {
        return;
    }
    chrome.tabs.create({
        url: chrome.extension.getURL('welcome.html'),
        active: true
    });
});

function vidOff() {
    vid.pause();
    vid.src = "";
    if (vid.srcObject) vid.srcObject.getTracks()[0].stop();
}

let infer = false;
// Get previously-stored infer checkbox setting, if any.
chrome.storage.local.get('infer', items => {
    infer = !!items['infer'];
});

// Listener for commands from the extension popup (controller) page.

let createtab = true;
// Setup webcam, initialize the KNN classifier model and start the work loop.
async function setupCam() {
    navigator.mediaDevices.getUserMedia({
        video: true
    }).then(mediaStream => {
        vid.srcObject = mediaStream;
    }).catch((error) => {
        console.warn(error);
    });
}

// If cam acecss gets granted to this extension, setup webcam.
chrome.storage.onChanged.addListener((changes, namespace) => {
    if ('infer' in changes) {
        var infer_state = changes['infer'].newValue;
        if (infer_state) {
            // If cam acecss has already been granted to this extension, setup webcam.
            chrome.storage.local.set({
                'camAccess': true
            }, () => {});

            chrome.storage.local.get('camAccess', items => {
                if (!!items['camAccess']) {
                    console.log('cam access already exists');
                    setupCam();
                }
            });

            console.log("ON");
            console.log(intervalId);

            if (intervalId) {
                clearInterval(intervalId);
            }
            intervalId = setInterval(() => {
                imageSrc = getImage();
                handleSubmit(imageSrc);
            }, 2000);
        } else {
            vidOff();
            if (intervalId) {
                clearInterval(intervalId);
            }
            chrome.storage.local.set({
                'camAccess': false
            }, () => {});
            label = "no_gesture"
            console.log('OFF');
        }

    }

});
const canvas = document.createElement("canvas");

var getImage = function() {
    let width = 227, height = 170.25;
    canvas.setAttribute('width', `${width}`); // clears the canvas
    canvas.setAttribute('height', `${height}`); // clears the canvas
    canvas.getContext('2d').drawImage(vid, 0, 0, width, height);
    var img = document.createElement("img");
    img.src = canvas.toDataURL();
    // image.innerHTML = '<img src="' + img.src + '"/>';
    return img.src
};

function handleSubmit(imageSrc) {
    // event.preventDefault();
    var url = "http://localhost:8000/movie_controller";
    var json = { imageSrc: imageSrc };
    json = JSON.stringify(json);

    fetch(url, {
        method: 'POST',
        body: json,
        headers: { 'Content-Type': 'application/json', "Access-Control-Allow-Origin": "*" },
        crossDomain: true
    }).then(res => res.json()).then(result => {
        var str = result['content'];
        label = str;
        if (str == "stop") {
            playStopVid();
            label = `${str} (stop/continue)`
        } else if (str == "ok") {
            // next
            next();
            label = `${str} (Next Video)`
        } else if (str == "mute") {
            mute()
            label = `${str} (Mute/Unmute)`
        } else if (str == "like") {
            volumeUpVid();
            label = `${str} (volume up)`
        } else if (str == "dislike") {
            volumeDownVid();
            label = `${str} (volume down)`
        } else if (str == "palm") {
            // resize
            sizeScreen();
            label = `${str} (resize screen)`
        } else if (str == "two_up") {
            speedUp();
            label = `${str} (speed up)`
        } else if (str == "peace") {
            speedDown();
            label = `${str} (speed down)`
        }
        sendMessage();
    })
}



const youtube = 'document.getElementsByClassName("video-stream html5-main-video")[0]';

function playStopVid() {
    chrome.tabs.executeScript({
        code: 'if (' + youtube + '.paused) { ' + youtube + '.play();}else {' + youtube + '.pause(); }',
    });
}

function volumeUpVid() {
    chrome.tabs.executeScript({
        code: youtube + '.volume = ' + youtube + '.volume > 0.9 ? 1 : ' + youtube + '.volume + 0.1 ',
    });
}

function mute() {
    chrome.tabs.executeScript({
        code: `document.getElementsByClassName("ytp-mute-button ytp-button")[0].click()`
    });
}

function volumeDownVid() {
    chrome.tabs.executeScript({
        code: youtube + '.volume = ' + youtube + '.volume < 0.1 ? ' + youtube + '.volume: ' + youtube + '.volume - 0.1 ',
    });
}

function next() {
    chrome.tabs.executeScript({
        code: `document.getElementsByClassName("ytp-next-button ytp-button")[0].click()`
    });
}

function seekVid() {
    chrome.tabs.executeScript({
        code: 'var player = ' + youtube + ';var curT = player.currentTime;player.currentTime = curT < player.duration - 10 ? curT + 10 : player.duration ',
    });
}

function rewindVid() {
    chrome.tabs.executeScript({
        code: 'var player = ' + youtube + ';var curT = player.currentTime;player.currentTime = curT < 10 ? 0 : curT - 10',
    });
}

function speedUp() {
    chrome.tabs.executeScript({
        code: youtube + '.playbackRate = 2.0'
    });
}

function speedDown() {
    chrome.tabs.executeScript({
        code: youtube + '.playbackRate = 1.0'
    });
}

function fullScreen() {
    sizeScreen();
    chrome.tabs.executeScript({
        code: `document.getElementsByClassName("ytp-fullscreen-button ytp-button")[0].click()`
    });
}

function sizeScreen() {
    chrome.tabs.executeScript({
        code: `document.getElementsByClassName("ytp-size-button ytp-button")[0].click()`
    });
}

function sendMessage() {
    chrome.runtime.sendMessage({data: label});
}

// setInterval(sendMessage, 100);