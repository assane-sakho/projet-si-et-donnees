import React from 'react';
import { Row, Col, Accordion } from 'react-bootstrap';
import GuessClothCategory from './GuessClothCategory';
import GuessClothInfos from './GuessClothInfos';
import GuessClothPrice from './GuessClothPrice';

function Features() {
    return (
        <Row className="mt-5 mb-5">
            <Col md={{ span: 6, offset: 3 }}>
                <Accordion defaultActiveKey={'0'}>
                    <GuessClothCategory />
                    <GuessClothInfos />
                    <GuessClothPrice />
                </Accordion>
            </Col>
        </Row>
    );
}

export default Features;
