from flask import Flask
from flask import send_from_directory

app = Flask(__name__,
            static_url_path='',
            static_folder='web/static', )


@app.route('/yatzy')
def send_file():
    return send_from_directory('web/static', 'hello.html')


app.run(host='0.0.0.0', port=8080)
