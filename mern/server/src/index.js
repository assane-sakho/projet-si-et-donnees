// server.js
import bodyParser from 'body-parser';
import express from 'express';
import path from 'path';
import db from './config/mongodb.config.js';
import request from "request"

// import postRouter from './routes/post.router';

const app = express();
const PORT = process.env.PORT || 8080;
const ML_API_URL = process.env.ML_API_URL || 'http://projet_si_et_donnes_ml:5000/';
const dirname = path.resolve();

// Routes
//const postRouter = require('./routes/post.router.js');

const CLIENT_BUILD_PATH = path.join(dirname, "../../client/build");

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
  console.log(`${ML_API_URL}/guessClotheType/`)
  
  request({
    method: 'POST',
    uri: `${ML_API_URL}/guessClotheType/`,
  }, function (error, response, body) {
    const data = response.body;
    res.send(data);
  });

})

app.listen(PORT, function () {
    console.log(`Server Listening on ${PORT}`);
});

export default app;
