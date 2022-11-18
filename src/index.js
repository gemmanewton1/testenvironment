import "./styles.css";
import * as tf from "@tensorflow/tfjs";
// import test from "./screenshot.jpg";
import test from "./test.jpg";

let classesDir = {
  1: { name: "Kangaroo", id: 1 },
  2: { name: "Others", id: 2 }
};
// let classesDir = {
//   1: { name: "Buttons", id: 1 },
//   2: { name: "Checkboxes", id: 2 },
//   3: { name: "FAB -Floating Action Button-", id: 3 },
//   4: { name: "Page Controls", id: 4 },
//   5: { name: "Pickers", id: 5 },
//   6: { name: "Progress indicators", id: 6 },
//   7: { name: "Radio buttons", id: 7 },
//   8: { name: "Rating", id: 8 },
//   9: { name: "Sliders", id: 9 },
//   10: { name: "Steppers", id: 10 },
//   11: { name: "Switches", id: 11 },
//   12: { name: "Text Fields", id: 12 }
// };

const loadImage = (frame) => {
  console.log("Pre-processing image...");
  const tfimg = tf.browser.fromPixels(frame).toInt();
  const expandedimg = tfimg.expandDims();
  return expandedimg;
};

const predict = async (inputs, model) => {
  console.log("Running predictions...");
  const predictions = await model.executeAsync(inputs);
  return predictions;
};

const renderPredictions = (predictions, width, height) => {
  console.log("Highlighting results...");

  //Getting predictions
  const boxes = predictions[4].arraySync();
  const scores = predictions[5].arraySync();
  const classes = predictions[6].dataSync();

  const detectionObjects = [];

  scores[0].forEach((score, i) => {
    if (score > 0.5) {
      const bbox = [];
      const minY = boxes[0][i][0] * height;
      const minX = boxes[0][i][1] * width;
      const maxY = boxes[0][i][2] * height;
      const maxX = boxes[0][i][3] * width;
      bbox[0] = minX;
      bbox[1] = minY;
      bbox[2] = maxX - minX;
      bbox[3] = maxY - minY;

      detectionObjects.push({
        class: classes[i],
        label: classesDir[classes[i]].name,
        score: score.toFixed(4),
        bbox: bbox
      });
    }
  });

  return detectionObjects;
};

const run = async () => {
  try {
    const image = document.getElementById("image");
    image.crossOrigin = "anonymous";
    image.src = test;

    const c = document.getElementById("canvas");
    c.width = image.width;
    c.height = image.height;
    const context = c.getContext("2d");
    context.drawImage(image, 0, 0);

    // Font options.
    const font = "16px sans-serif";
    context.font = font;
    context.textBaseline = "top";

    const model = await tf.loadGraphModel(
      "https://raw.githubusercontent.com/hugozanini/TFJS-object-detection/master/models/kangaroo-detector/model.json"
      // "https://raw.githubusercontent.com/dusskapark/zeplin-ml/main/public/models/web_model/model.json"
    );

    const expandedimg = loadImage(image);
    const predictions = await predict(expandedimg, model);
    const detections = renderPredictions(
      predictions,
      image.width,
      image.height
    );
    console.log(predictions);
    console.log("detected: ", detections);

    detections.forEach((item) => {
      const x = item["bbox"][0];
      const y = item["bbox"][1];
      const width = item["bbox"][2];
      const height = item["bbox"][3];

      // Draw the bounding box.
      context.strokeStyle = "#00FFFF";
      context.lineWidth = 4;
      context.strokeRect(x, y, width, height);

      // Draw the label background.
      context.fillStyle = "#00FFFF";
      const textWidth = context.measureText(
        item["label"] + " " + (100 * item["score"]).toFixed(2) + "%"
      ).width;
      const textHeight = parseInt(font, 10); // base 10
      context.fillRect(x, y, textWidth + 4, textHeight + 4);
    });

    for (let i = 0; i < detections.length; i++) {
      const item = detections[i];
      const x = item["bbox"][0];
      const y = item["bbox"][1];
      const content =
        item["label"] + " " + (100 * item["score"]).toFixed(2) + "%";

      // Draw the text last to ensure it's on top.
      context.fillStyle = "#000000";
      context.fillText(content, x, y);
      console.log("detected: ", item);
    }
  } catch (e) {
    console.log(e.message);
  }
};

if (document.readyState !== "loading") {
  run();
} else {
  document.addEventListener("DOMContentLoaded", run);
}
