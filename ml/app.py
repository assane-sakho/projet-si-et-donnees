from flask import Flask, flash, request, redirect, url_for
import os

app = Flask(__name__)
port = int(os.environ.get('PORT', 5000))

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route("/guessClotheType/", methods=['POST'])
def guess_cloth_type():
    if request.method == 'POST':
        # if 'file' not in request.files:
        #     flash('No file part')
        #     return redirect(request.url)
        # file = request.files['file']
        # # if user does not select file, browser also
        # # submit a empty part without filename
        # if file.filename == '':
        #     flash('No selected file')
        #     return redirect(request.url)       
        # if file and allowed_file(file.filename):
            # return "t-shirt"
        return "t-shirt"

def allowed_file(filename):     
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, debug=True)
