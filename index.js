var express = require('express');
var app = express();
var database = require('./database.json');

app.listen(80, function () {
	console.log('server running on port 80');
})

// Predict image
const multer = require('multer')
var storage = multer.diskStorage({
	destination: "images",
	filename: function (req, file, cb) {
		cb(null, Date.now() + '.jpg') //Appending .jpg
	}
})

var upload = multer({
	storage: storage
});

function predictImage(req, res) {
	imagepath = 'images/' + req.file.filename
	var spawn = require('child_process').spawn;
	var process = spawn('python3', ['./predict.py', imagepath]);

	process.stdout.on('data', function (data) {
		let name = data.toString().replace('_', ' ');
		name = name.replace('\n', '');
		let nutritions = database[name];
		let result = {
			name: name,
			nutritions: nutritions
		}
		res.send(JSON.stringify(result));
	})
}

app.get('/', function (req, res) {
	res.send('Hello')
})
app.post('/predict', upload.single('upload'), predictImage);
