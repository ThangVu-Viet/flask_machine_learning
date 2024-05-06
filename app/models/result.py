from app.extensions import db


class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Real_result = db.Column(db.Float)
    PL_regression = db.Column(db.Float)
    Gradient_boosting = db.Column(db.Float)
    RNN = db.Column(db.Float)

    def __init__(self, Real_result, PL_regression, Gradient_boosting, RNN):
        self.Real_result = Real_result
        self.PL_regression = PL_regression
        self.Gradient_boosting = Gradient_boosting
        self.RNN = RNN
