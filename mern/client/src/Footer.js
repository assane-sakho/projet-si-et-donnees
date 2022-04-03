import React from 'react';
import { Row, Col } from 'react-bootstrap';

function Footer() {
  return (
    <Row>
      <Col>
        <hr />
        <p>
          You can check further in information on this project here {' '}
          <a
            href="https://github.com/assane-sakho/projet-si-et-donnees"
            target="_blank"
            rel="noopener noreferrer"
          >
            here
          </a>
          .
        </p>
      </Col>
    </Row>

  );
}

export default Footer;
