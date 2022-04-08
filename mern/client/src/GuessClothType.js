import { useState, useEffect, React } from 'react';
import { Button, Form, Alert, Accordion, Spinner } from 'react-bootstrap';
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
    const [picture, setPicture] = useState(undefined);
    const [type, setType] = useState('');
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (picture !== undefined) {
            setLoading(true);
            const data = new FormData()

            data.append('file', picture)
            axios.post("/api/cloth_type/predict", data)
                .then((res) => {
                    setLoading(false);
                    setType(res.data)
                });
        }

    }, [picture]);

    return (
        <Accordion.Item eventKey="0">
            <Accordion.Header>#1 : Deviner le type de vêtement</Accordion.Header>
            <Accordion.Body>
                <AlertInstruction />

                <Form.Group controlId="guessClothTypeInput" className="mb-3 mt-3">
                    <Form.Control type="file" accept=".png,.jpg,.jpeg" onChange={(e) => setPicture(e.target.files[0])} />
                </Form.Group>
                {loading ? <Spinner animation="border" /> : <></>}
                {type !== '' ?

                    <p>
                        Le type de vêtement correspond à <b className="text-success">{type}.</b>
                    </p>
                    : <></>
                }

            </Accordion.Body>
        </Accordion.Item>
    );
}

export default GuessClothType;
