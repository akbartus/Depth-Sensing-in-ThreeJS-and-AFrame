/* Depth sensing using THREE.js */
const w = 300, h = 300;
const video = document.getElementById("video");
[video.width, video.height] = [w, h];
let sceneCleared = false;
let model;

const constraints = {
  audio: false,
  video: { width: w, height: h,  facingMode: "environment" }
};
navigator.mediaDevices.getUserMedia(constraints)
  .then((mediaStream) => {
    const video = document.querySelector("video");
    video.srcObject = mediaStream;
    video.onloadedmetadata = () => {
      video.play();
    };
  })
  .catch((err) => {
    console.error(`${err.name}: ${err.message}`);
  });


const load = async () => {
  const url = "./224x224_fp32/model.json";
  model = await tf.loadGraphModel(url);
  console.log("Model loaded");
  tf.setBackend('wasm'); // wasm or webgl : https://www.tensorflow.org/js/guide/platform_environment. Wasm is good choice for lower end devices. Webgl is good for powerful ones.
  tf.enableProdMode(); // enables production mode, removes NaN check
};
load();

let canvas = document.querySelector("#canvas");

document.querySelector("#getDepth").addEventListener("click",
function predict() {
  // prevent memory leakage using tf.tidy();
  tf.tidy(() => {
    const image = tf.browser
      .fromPixels(video)
      .resizeBilinear([224, 224])
      .asType("float32")
      .div(255);
    const batchedImg = image.expandDims(0);
    const result = model.predict(batchedImg);

    const outReshape = tf.reshape(result, [224, 224, 1]);
    const outResize = tf
      .mul(tf.div(outReshape, tf.max(outReshape)), 255)
      .asType("int32");
    tf.browser.toPixels(outResize, canvas);
  });
  requestAnimationFrame(predict);

  console.log(tf.memory());
})

function getImageData(imgOrCanvas) {
  const canvas = document.createElement("canvas");
  canvas.width = imgOrCanvas.width;
  canvas.height = imgOrCanvas.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgOrCanvas, 0, 0);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

function getPixel(imageData, u, v) {
  const x = (u * (imageData.width - 1)) | 0;
  const y = (v * (imageData.height - 1)) | 0;
  if (x < 0 || x >= imageData.width || y < 0 || y >= imageData.height) {
    return [0, 0, 0, 0];
  } else {
    const offset = (y * imageData.width + x) * 4;
    return Array.from(imageData.data.slice(offset, offset + 4)).map(
      (v) => v / 255
    );
  }
}

async function main() {
  const video = document.querySelector("video");
  const depthCanvas = document.getElementById("canvas2");
  const renderer = new THREE.WebGLRenderer({ canvas: depthCanvas, antialias: true });
  renderer.setClearColor(0xffffff);
  const fov = 45;
  const aspect = 0.5; // desktop aspect ratio 2/1
  const near = 1;
  const far = 100;
  const camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
  camera.position.z = 3.5;
  const controls = new THREE.OrbitControls(camera, depthCanvas);
  controls.target.set(0, 0, 0);
  controls.update();
  const scene = new THREE.Scene();
  
 let prevPositions = []; // Store the positions from the previous frame

function updateDepthCanvas() {
  if (sceneCleared == false) {
    scene.clear();
    sceneCleared == true;
  }
  const rgbData = getImageData(video);
  const depthData = getImageData(document.querySelector("#canvas"));

  const skip = 2;
  const across = Math.ceil(rgbData.width / skip);
  const down = Math.ceil(rgbData.height / skip);

  const positions = [];
  const colors = [];
  const color = new THREE.Color();
  const spread = 1; // Number of elements
  const depthSpread = 1; // Depth, the higher difference, the more
  const imageAspect = rgbData.width / rgbData.height;

  let pointCount = 0; // Variable to keep track of the number of points added

  for (let y = 0; y < down; ++y) {
    const v = y / (down - 1);
    for (let x = 0; x < across; ++x) {
      const u = x / (across - 1);
      const rgb = getPixel(rgbData, u, v);
      const depth = 1 - getPixel(depthData, u, v)[0];
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
    const smoothingFactor = 0.5; // Adjust the smoothing factor as needed
    for (let i = 0; i < positions.length; i += 3) {
      positions[i] = smoothingFactor * positions[i] + (1 - smoothingFactor) * prevPositions[i];
      positions[i + 1] = smoothingFactor * positions[i + 1] + (1 - smoothingFactor) * prevPositions[i + 1];
      positions[i + 2] = smoothingFactor * positions[i + 2] + (1 - smoothingFactor) * prevPositions[i + 2];
    }
  }

  prevPositions = positions.slice(); // Store the current positions for the next frame

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3)
  );
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const vertexShader = `
    attribute vec3 color;
    varying vec3 vColor;
    void main() {
      vColor = color;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = 4.0;
    }
  `;
  const fragmentShader = `
    varying vec3 vColor;

    void main() {
      gl_FragColor = vec4(vColor, 1.0);
    }
  `;
  const material = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
  });

  const points = new THREE.Points(geometry, material);

  scene.add(points);

  renderer.render(scene, camera);
  requestAnimationFrame(updateDepthCanvas);
}
  updateDepthCanvas();
}
main();
