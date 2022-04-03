import { useState, useEffect, React } from 'react';
import { Button, Form, Alert, Accordion } from 'react-bootstrap';
import axios from "axios"

function AlertInstruction() {
    const [show, setShow] = useState(true);

    if (show) {
        return (
            <Alert variant="info" onClose={() => setShow(false)} dismissible>
                <Alert.Heading>
                    Instruction
                </Alert.Heading>
                <p>
                    Envoyez une photo d'un vêtement pour que l'algorithme de Machine Learning détecte son type (pull, t-shirt, robe, etc).
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

function GuessClothType() {
    const [picture, setPicture] = useState(null);

    useEffect(() => {
        const data = new FormData()
        data.append('file', picture)
        axios.post("/api/guess_cloth_type", data, { 

       })

    }, [picture]);

    return (
        <Accordion.Item eventKey="0">
            <Accordion.Header>#1 : Deviner le type de vêtement</Accordion.Header>
            <Accordion.Body>
                <AlertInstruction />

                <Form.Group controlId="guessClothTypeInput" className="mb-3">
                    <Form.Control type="file" accept=".png,.jpg,.jpeg" onChange={(e) => setPicture(e.target.files[0])} />
                </Form.Group>
            </Accordion.Body>
        </Accordion.Item>
    );
}

export default GuessClothType;
