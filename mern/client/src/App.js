import React from 'react';
import './App.css';
import { Container } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import Header from './Header';
import Footer from './Footer';
import Features from './Features';
import axios from "axios"

function App() {
  if(process.env.REACT_APP_API_URL != undefined)
  {
    axios.defaults.baseURL = process.env.REACT_APP_API_URL;
  }

  return (
    <Container fluid className="bg-grey">
      <Header />

      <Features />

      <Footer />
    </Container >

  );
}

export default App;
