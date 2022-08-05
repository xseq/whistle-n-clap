# formatting time in YYYYMMDD_HHMM format

from datetime import datetime


def time_to_text():
    now = datetime.now()
    time_text = now.strftime('%Y%m%d_%H%M')
    return time_text

