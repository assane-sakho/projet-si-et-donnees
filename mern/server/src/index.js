// server.js
import bodyParser from 'body-parser';
import express from 'express';
import fileUpload from 'express-fileupload';
import path from 'path';
import db from './config/mongodb.config.js';
import axios from "axios"
import FormData from 'form-data';

// import postRouter from './routes/post.router';

const app = express();
const PORT = process.env.PORT || 8080;
const ML_API_URL = process.env.ML_API_URL || 'http://projet_si_et_donnes_ml:5000';
const dirname = path.resolve();

// Routes
//const postRouter = require('./routes/post.router.js');

const CLIENT_BUILD_PATH = path.join(dirname, "../client/build");

app.use(fileUpload());

app.use(
  bodyParser.urlencoded({
    extended: true
  })
);
app.use(bodyParser.json());

//  Route for client
app.use(express.static(CLIENT_BUILD_PATH));

// Server API's
// app.use('/api/posts', postRouter);

// Server React Client
app.get("/", function (req, res) {
  res.sendFile(path.join(CLIENT_BUILD_PATH, "index.html"));
});

app.post('/api/guess_cloth_type', function (req, res) {
  const file = req.files.file;
  let d = new FormData();
  d.append('file', file.data, {
    contentType: file.mimetype,
    filename: file.name
  });

  return axios.post(`${ML_API_URL}/guessClotheType/`, d, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }).catch((err) => {
    console.log(err)
    res.status(500).send(err);
  }).then((response) => {
    console.log('pye')
    res.send(response.data);
  });
})

app.listen(PORT, function () {
  console.log(`Server Listening on ${PORT}`);
});

export default app;
