import {MnistData} from './data.js';
var canvas, ctx, saveButton, clearButton;
var pos = {x:0, y:0};
var rawImage;
var model;
	
function getModel() {
	model = tf.sequential();

	// kernelSize: define the size of convolutional filter -> 3x3 filters
	// filters: define a set of filters to learn convolutions -> 8 filters
	// 
	model.add(tf.layers.conv2d({inputShape: [28, 28, 1], kernelSize: 3, filters: 8, activation: 'relu'}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
	model.add(tf.layers.conv2d({filters: 16, kernelSize: 3, activation: 'relu'}));
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
	model.add(tf.layers.flatten());
	model.add(tf.layers.dense({units: 128, activation: 'relu'}));
	model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

	// compile the model
	model.compile({optimizer: tf.train.adam(), loss: 'categoricalCrossentropy', metrics: ['accuracy']});

	return model;
}

async function train(model, data) {
	// define a set of metrics that sould tracks
	const metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy'];
	// define a container yg akan merender feedback  (just set a name and any required styles)
	const container = { name: 'Model Training', styles: { height: '640px' } };
	// here is the built in callbacks using the tf-visualization library, need to pass a container and the metrics to show the container with metrics
	const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
  
	const BATCH_SIZE = 512;
	const TRAIN_DATA_SIZE = 5500;
	const TEST_DATA_SIZE = 1000;

	const [trainXs, trainYs] = tf.tidy(() => {
		const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
		return [
			d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
			d.labels
		];
	});

	const [testXs, testYs] = tf.tidy(() => {
		const d = data.nextTestBatch(TEST_DATA_SIZE);
		return [
			d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
			d.labels
		];
	});

	// train the model
	return model.fit(trainXs, trainYs, {
		batchSize: BATCH_SIZE,
		validationData: [testXs, testYs],	// set of validation data to validate the model and report back in accuracy
		epochs: 20,
		shuffle: true,	// randomize the data for training to prevent overfitting cause of multiple similar classes in the same batch
		callbacks: fitCallbacks
	});
}

function setPosition(e){
	pos.x = e.clientX-100;
	pos.y = e.clientY-100;
}
    
function draw(e) {
	if(e.buttons!=1) return;
	ctx.beginPath();
	ctx.lineWidth = 24;
	ctx.lineCap = 'round';
	ctx.strokeStyle = 'white';
	ctx.moveTo(pos.x, pos.y);
	setPosition(e);
	ctx.lineTo(pos.x, pos.y);
	ctx.stroke();
	rawImage.src = canvas.toDataURL('image/png');
}
    
function erase() {
	ctx.fillStyle = "black";
	ctx.fillRect(0,0,280,280);
}
    
function save() {
	var raw = tf.browser.fromPixels(rawImage,1);
	var resized = tf.image.resizeBilinear(raw, [28,28]);
	var tensor = resized.expandDims(0);
  var prediction = model.predict(tensor);
  var pIndex = tf.argMax(prediction, 1).dataSync();
    
	alert(pIndex);
}
    
// init function to set up the UI
function init() {
	canvas = document.getElementById('canvas');
	rawImage = document.getElementById('canvasimg');
	ctx = canvas.getContext("2d");
	ctx.fillStyle = "black";
	ctx.fillRect(0,0,280,280);
	canvas.addEventListener("mousemove", draw);
	canvas.addEventListener("mousedown", setPosition);
	canvas.addEventListener("mouseenter", setPosition);
	saveButton = document.getElementById('sb');
	saveButton.addEventListener("click", save);
	clearButton = document.getElementById('cb');
	clearButton.addEventListener("click", erase);
}


async function run() {  
	const data = new MnistData();		// calling the class that created in data.js
	await data.load();							// download the sprite and images and get them into memory
	const model = getModel();				// build the model and return to this model object
	tfvis.show.modelSummary({name: 'Model Architecture'}, model);		// showing model architecture
	await train(model, data);				// train the model
	init();													// calling the init function
	alert("Training is done, try classifying your handwriting!");
}

document.addEventListener('DOMContentLoaded', run);



    
