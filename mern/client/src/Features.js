import React from 'react';
import { Row, Col, Accordion } from 'react-bootstrap';
import GuessClothType from './GuessClothType';

function Features() {
    return (
        <Row className="mt-5 mb-5">
            <Col md={{ span: 6, offset: 3 }}>
                <Accordion defaultActiveKey={['0']} alwaysOpen>
                    <GuessClothType />
                </Accordion>
            </Col>
        </Row>
    );
}

export default Features;
