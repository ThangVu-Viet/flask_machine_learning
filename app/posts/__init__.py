import random

from flask import Blueprint

from app.models.post import Post

bp = Blueprint('posts', __name__)

from app.posts import routes

