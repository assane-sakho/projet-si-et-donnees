import React from 'react';
import { Row, Col, Card } from 'react-bootstrap';
import Emoji from './Emoji';

function Header() {
  return (
    <Row className="mt-5 mb-5">
      <Col>
        <div className="d-flex justify-content-center">
          <Card border="primary">
            <Card.Header>
              Welcome To Clothers 
              <Emoji symbol="👕" label="t-shirt" />
              <Emoji symbol="👖" label="t-shirt" />
              <Emoji symbol="👔" label="t-shirt" />
              <Emoji symbol="👗" label="t-shirt" />
              <Emoji symbol="👘" label="t-shirt" />
              <Emoji symbol="🧥" label="t-shirt" />
            </Card.Header>
          </Card>
        </div>

      </Col>
    </Row>

  );
}

export default Header;
