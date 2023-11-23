AFRAME.registerComponent('depth-sensing', {
    schema: {
        modelSrc: { type: 'string', default: "224x224_fp32/model.json"},
        tensorWidth: { type: "number", default: 224 },
        tensorHeight: { type: 'number', default: 224 },
        pixelSkipNumber: { type: 'number', default: 2 },
        spreadValue: { type: 'number', default: 1 },
        depthValue: { type: 'number', default: 1 },
        smoothingFactor: { type: 'string', default: "0.5" },
        pointSize: { type: 'string', default: "8.0" }
    },
    init: function () {
        let mSrc = this.data.modelSrc;
        let tWidth = this.data.tensorWidth;
        let tHeight = this.data.tensorHeight;
        let pSkipNumber = this.data.pixelSkipNumber;
        let sValue = this.data.spreadValue;
        let dValue = this.data.depthValue;
        let sFactor = this.data.smoothingFactor;
        let pSize = this.data.pointSize;
        //Create Necessary html elements
        let video = document.createElement('video');
        video.id = 'video';
        video.autoplay = true;
        video.loop = true;
        video.muted = true;
        video.playsinline = true;
        video.style.display = 'none';

        let canvas = document.createElement('canvas');
        canvas.id = 'canvas';
        canvas.setAttribute("style", "position: absolute;bottom: 20px;left: 20px;border: 1px solid #000;background-color: rgb(250, 244, 250);z-index: 2;width: 100px;height: 100px;padding: 0;")
        
        let button = document.createElement('button');
        button.id = 'getDepth';
        button.textContent = 'Get Depth';
        button.setAttribute("style", "position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 3;");
        // Append elements to the body or any other container element
        document.body.appendChild(video);
        document.body.appendChild(canvas);
        document.body.appendChild(button);

        /* Depth sensing code */
        let w = 300, h = 300;
        [video.width, video.height] = [w, h];

        let model;
        let letraints = {
            audio: false,
            video: { width: w, height: h, facingMode: "environment" }
        };
        navigator.mediaDevices.getUserMedia(letraints)
            .then((mediaStream) => {
                let video = document.querySelector("video");
                video.srcObject = mediaStream;
                video.onloadedmetadata = () => {
                    video.play();
                };
            })
            .catch((err) => {
                console.error(`${err.name}: ${err.message}`);
            });


        let load = async () => {
            let url = mSrc;
            model = await tf.loadGraphModel(url);
            console.log("Model loaded");
            tf.setBackend('wasm'); // wasm or webgl
            tf.enableProdMode();
        };
        load();


        document.querySelector("#getDepth").addEventListener("click",
            function predict() {
                tf.tidy(() => {
                    let image = tf.browser
                        .fromPixels(video)
                        .resizeBilinear([tWidth, tHeight])
                        .asType("float32")
                        .div(255);
                    let batchedImg = image.expandDims(0);
                    let result = model.predict(batchedImg);

                    let outReshape = tf.reshape(result, [tWidth, tHeight, 1]);
                    let outResize = tf
                        .mul(tf.div(outReshape, tf.max(outReshape)), 255)
                        .asType("int32");
                    tf.browser.toPixels(outResize, canvas);
                });
                requestAnimationFrame(predict);
            })

        function getImageData(imgOrCanvas) {
            let canvas = document.createElement("canvas");
            canvas.width = imgOrCanvas.width;
            canvas.height = imgOrCanvas.height;
            let ctx = canvas.getContext("2d");
            ctx.drawImage(imgOrCanvas, 0, 0);
            return ctx.getImageData(0, 0, canvas.width, canvas.height);
        }

        function getPixel(imageData, u, v) {
            let x = (u * (imageData.width - 1)) | 0;
            let y = (v * (imageData.height - 1)) | 0;
            if (x < 0 || x >= imageData.width || y < 0 || y >= imageData.height) {
                return [0, 0, 0, 0];
            } else {
                let offset = (y * imageData.width + x) * 4;
                return Array.from(imageData.data.slice(offset, offset + 4)).map(
                    (v) => v / 255
                );
            }
        }

        async function main() {
            let video = document.querySelector("video");
            let scene = document.querySelector("a-scene").object3D;
            let thisElement = document.querySelector("a-entity[depth-sensing]").object3D;
            let renderer = document.querySelector("a-scene").renderer;

            let prevPositions = []; // Store the positions from the previous frame

            function updateDepthCanvas() {

                let oldPoints = thisElement.children.filter(child => child instanceof THREE.Points);
                oldPoints.forEach(point => {
                    thisElement.remove(point);
                    point.geometry.dispose();
                    point.material.dispose();
                });
                let rgbData = getImageData(video);
                let depthData = getImageData(document.querySelector("#canvas"));

                let skip = pSkipNumber;
                let across = Math.ceil(rgbData.width / skip);
                let down = Math.ceil(rgbData.height / skip);

                let positions = [];
                let colors = [];
                let color = new THREE.Color();
                let spread = sValue;
                let depthSpread = dValue;
                let imageAspect = rgbData.width / rgbData.height;

                let pointCount = 0;

                for (let y = 0; y < down; ++y) {
                    let v = y / (down - 1);
                    for (let x = 0; x < across; ++x) {
                        let u = x / (across - 1);
                        let rgb = getPixel(rgbData, u, v);
                        let depth = 1 - getPixel(depthData, u, v)[0];
                        positions.push(
                            (u * 2 - 1) * spread * imageAspect,
                            (v * -2 + 1) * spread,
                            depth * depthSpread
                        );
                        colors.push(...rgb.slice(0, 3));
                        pointCount++;
                    }
                }

                // Apply temporal filtering to smooth the movement
                if (prevPositions.length > 0) {
                    let smoothingFactor = sFactor;
                    for (let i = 0; i < positions.length; i += 3) {
                        positions[i] = smoothingFactor * positions[i] + (1 - smoothingFactor) * prevPositions[i];
                        positions[i + 1] = smoothingFactor * positions[i + 1] + (1 - smoothingFactor) * prevPositions[i + 1];
                        positions[i + 2] = smoothingFactor * positions[i + 2] + (1 - smoothingFactor) * prevPositions[i + 2];
                    }
                }

                prevPositions = positions.slice();
                let geometry = new THREE.BufferGeometry();
                geometry.setAttribute(
                    "position",
                    new THREE.Float32BufferAttribute(positions, 3)
                );
                geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
                let vertexShader = `
                attribute vec3 color;
                varying vec3 vColor;
                void main() {
                vColor = color;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                gl_PointSize = ${pSize};
                }`;
                let fragmentShader = `
                varying vec3 vColor;
                void main() {
                gl_FragColor = vec4(vColor, 1.0);
                }`;
                let material = new THREE.ShaderMaterial({
                    vertexShader,
                    fragmentShader,
                });
                let points = new THREE.Points(geometry, material);
                thisElement.add(points);
                requestAnimationFrame(updateDepthCanvas);
            }
            updateDepthCanvas();
        }
        main();
    }
});