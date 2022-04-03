import React from 'react';
import { Row, Col, Card } from 'react-bootstrap';

function Header() {
  return (
    <Row className="mt-5 mb-5">
      <Col>
        <div className="d-flex justify-content-center">
          <Card border="primary" style={{ width: '32rem' }}>
            <Card.Header>Welcome To Clothers</Card.Header>
          </Card>
        </div>

      </Col>
    </Row>

  );
}

export default Header;
