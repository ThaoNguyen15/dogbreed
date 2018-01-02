from os.path import dirname, join, abspath
from flask import ( Flask, request, redirect, url_for,
                    render_template, send_from_directory )
from werkzeug.utils import secure_filename
from argparse import ArgumentParser

UPLOAD_FOLDER = join(dirname(abspath(__file__)), 'user_images')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
DEFAULT_PORT = 8571

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(join(app.config['UPLOAD_FOLDER'], filename))
            img_path = join(app.config['UPLOAD_FOLDER'], filename)
            return render_template(
                'predict.html',
                img_path=url_for('send_file',filename=filename),
                dog_human=True,
                species='dog',
                dog_breed='italian_greyhound'
            )
    return render_template('predict.html')

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/test')
def show_img():
    return render_template(
        'predict.html',
        img_path=url_for('send_file',filename=filename),
        dog_human=True,
        species='dog',
        dog_breed='italian_greyhound'
    )
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--port', default=DEFAULT_PORT,
                        help='Port number to run')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode?')
    args = parser.parse_args()
    app.run(port=args.port, debug=args.debug)
