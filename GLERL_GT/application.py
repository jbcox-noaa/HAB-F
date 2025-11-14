# application.py

from cnn_model_dashboard import create_dash_app
from datetime import date, timedelta

# You can refactor this date block later into a helper if you want
START_DATE_1 = date(2024, 6, 29)
END_DATE_1   = date(2024, 7, 9)

START_DATE_2 = date(2025, 7, 25)
END_DATE_2   = date(2025, 8, 2)

span_1 = (END_DATE_1 - START_DATE_1).days
span_2 = (END_DATE_2 - START_DATE_2).days

dates = [START_DATE_1 + timedelta(days=i) for i in range(span_1 + 1)] + \
        [START_DATE_2 + timedelta(days=i) for i in range(span_2 + 1)]

app = create_dash_app(dates)
application = app.server
