from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))

def get_kst_now_iso():
    return datetime.now(KST).isoformat()
