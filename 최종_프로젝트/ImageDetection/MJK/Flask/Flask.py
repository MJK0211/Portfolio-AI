from flask import Flask

app = Flask(__name__)

@app.route('/')                 # 주소 뒤에 붙으면 해당 이름 함수내용이 실행된다
def home():
    return 'Hello, World!'

@app.route('/user')
def user():
    return 'Hello, User!'

if __name__ == '__main__':
    app.run(debug=True)

