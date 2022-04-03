import os

from waitress import serve
import app
serve(app.app, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
