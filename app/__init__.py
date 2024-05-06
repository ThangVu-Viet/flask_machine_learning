from flask import Flask
from config import Config
from app.extensions import db

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)

    from app.main import bp as main_bp
    app.register_blueprint(main_bp)

    from app.posts import bp as post_bp
    app.register_blueprint(post_bp, url_prefix='/posts')

    from app.page import bp as page_bp
    app.register_blueprint(page_bp, url_prefix='/page')

    from app.questions import bp as questions_bp
    app.register_blueprint(questions_bp, url_prefix='/questions')
    return app
