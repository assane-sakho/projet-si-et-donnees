// server.js
import bodyParser from 'body-parser';
import express from 'express';
import fileUpload from 'express-fileupload';
import path from 'path';
import db from './config/mongodb.config.js';
import axios from "axios"
import FormData from 'form-data';
import formidable from 'formidable';
import { PassThrough } from 'stream';

// import postRouter from './routes/post.router';

const app = express();
const PORT = process.env.PORT || 8080;
const ML_API_URL = process.env.ML_API_URL || 'http://projet_si_et_donnes_ml:5000';
const dirname = path.resolve();

// Routes
//const postRouter = require('./routes/post.router.js');

const CLIENT_BUILD_PATH = path.join(dirname, "../client/build");

// app.use(fileUpload());

// app.use(
// 	bodyParser.urlencoded({
// 	extended: true
// 	})
// );
// app.use(bodyParser.json());

app.use(express.json());
app.use(express.urlencoded());

//  Route for client
app.use(express.static(CLIENT_BUILD_PATH));

// Server API's
// app.use('/api/posts', postRouter);

// Server React Client
app.get("/", function (req, res) {
  	res.sendFile(path.join(CLIENT_BUILD_PATH, "index.html"));
});

app.post('/api/cloth_type/train', function (req, res) {
	return axios.post(`${ML_API_URL}/cloth_type/train/`)
				.catch((err) => {
					// console.log(err)
					res.status(500).send(err);
				}).then((response) => {
					console.log('/api/cloth_type/train')
					console.log(response.data)
					res.send(response.data);
				});
})

app.post('/api/cloth_type/predict', function (req, res) {

	const uploadStream = (file) => {
		var pass = PassThrough()
		const formData = new FormData();
		
		formData.append('flask_file_field', pass, file.originalFilename);
		formData.submit(`${ML_API_URL}/cloth_type/predict/`, (err, r) => {
			r.resume();
			var result = '';
			r.on('readable', () => {
				var tmp = r.read();
				if(tmp != undefined)
				{
					result += tmp;
				}
			})
			r.on('end', () => {
				res.send(result);
			})
		  });
		return pass;
	  };

	const form =  formidable({
		fileWriteStreamHandler: uploadStream
	  });

	  form.parse(req, (err, fields, files) => {
		//res.json('Success!');
	  });
})

app.listen(PORT, function () {
  	console.log(`Server Listening on ${PORT}`);
});

export default app;
