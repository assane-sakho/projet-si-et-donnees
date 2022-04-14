import { useState, useEffect, React } from 'react';
import { Button, Form, Alert, Accordion, Spinner, Image, Col, Carousel, Row } from 'react-bootstrap';
import axios from "axios"

function AlertInstruction() {
    const [show, setShow] = useState(true);

    if (show) {
        return (
            <Alert variant="success" onClose={() => setShow(false)} dismissible>
                <Alert.Heading>
                    Instruction
                </Alert.Heading>
                <p>
                    Envoyez une photo d'un vêtement pour que l'algorithme de Machine Learning détecte ses informations (matière, couleur, etc).
                </p>
            </Alert>
        );
    }
    return (
        <Button variant="info" onClick={() => setShow(true)}>
            Afficher l'instruction
        </Button>
    );
}

function GuessClothInfos() {
    const [picture, setPicture] = useState(undefined);
    const [file, setFile] = useState(undefined);
    const [type, setType] = useState('');
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (picture !== undefined) {
            setLoading(true);
            setFile(URL.createObjectURL(picture));
            const data = new FormData()

            data.append('file', picture)
            axios.post("/api/guess_cloth_type", data)
                .then((res) => {
                    setLoading(false);
                    setType(res.data)
                });
        }

    }, [picture]);

    return (
        <Accordion.Item eventKey="1">
            <Accordion.Header>#2 : Informations complémentaire d'un vêtement</Accordion.Header>
            <Accordion.Body>
                <Row>
                    <AlertInstruction />
                </Row>
                <Row>
                    <Form.Group controlId="guessClothTypeInput" className="mb-3 mt-3">
                        <Form.Control type="file" accept=".png,.jpg,.jpeg" onChange={(e) => setPicture(e.target.files[0])} />
                    </Form.Group>
                </Row>
                <Row>
                    <Col md={{ span: 6}}>
                        <Image src={file}  thumbnail />
                    </Col>
                    <Col md={{ span: 6}}>
                    <Carousel variant="dark">
                            <Carousel.Item>
                                <img
                                className="d-block w-100"
                                src="holder.js/800x400?text=First slide&bg=f5f5f5"
                                alt="First slide"
                                />
                                <Carousel.Caption>
                                <h5>First slide label</h5>
                                <p>Nulla vitae elit libero, a pharetra augue mollis interdum.</p>
                                </Carousel.Caption>
                            </Carousel.Item>
                            <Carousel.Item>
                                <img
                                className="d-block w-100"
                                src="holder.js/800x400?text=Second slide&bg=eee"
                                alt="Second slide"
                                />
                                <Carousel.Caption>
                                <h5>Second slide label</h5>
                                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
                                </Carousel.Caption>
                            </Carousel.Item>
                            <Carousel.Item>
                                <img
                                className="d-block w-100"
                                src="holder.js/800x400?text=Third slide&bg=e5e5e5"
                                alt="Third slide"
                                />
                                <Carousel.Caption>
                                <h5>Third slide label</h5>
                                <p>Praesent commodo cursus magna, vel scelerisque nisl consectetur.</p>
                                </Carousel.Caption>
                            </Carousel.Item>
                        </Carousel>
                    </Col>
                </Row>
                <Row>
                    {loading ? <Spinner animation="border" /> : <></>}
                    {type !== '' ?

                        <p>
                            Le type de vêtement correspond à <b className="text-success">{type}.</b>
                        </p>
                        : <></>
                    }
                </Row>
            </Accordion.Body>
        </Accordion.Item>
    );
}

export default GuessClothInfos;
