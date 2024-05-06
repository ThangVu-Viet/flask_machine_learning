from sqlalchemy import create_engine, text
import pandas as pd


def db_process(y_test, y_pred_rgr, y_pred_bst, y_pred_rnn):
    # Connect to the database
    engine = create_engine("mysql+mysqlconnector://root:123456@localhost/post294")

    df = pd.DataFrame({
        'Real_result': y_test.reshape(-1),
        'PL_regression': y_pred_rgr.reshape(-1),
        'Gradient_Boosting': y_pred_bst,
        'RNN': y_pred_rnn
    })
    # Clear value data in table before add new values
    with engine.connect() as conn:
        conn.exec_driver_sql("TRUNCATE TABLE result")
        conn.commit()

    df.to_sql(
        name='result',
        con=engine,
        if_exists='append',
        index=False,
    )
