import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from section2 import ASSETS


def main():
    parser = argparse.ArgumentParser(description="Download the 15 Yahoo Finance close series.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--translate-relay", action="store_true",
                        help="Use the Google Translate relay for the public chart endpoint.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    host = "query1.finance.yahoo.com"
    params = {"period1": 1577836800, "period2": 1735689600, "interval": "1d"}
    if args.translate_relay:
        host = "query1-finance-yahoo-com.translate.goog"
        params.update({"_x_tr_sl": "auto", "_x_tr_tl": "en", "_x_tr_hl": "en"})
    for code in ASSETS.values():
        url = f"https://{host}/v8/finance/chart/{code}%3DF?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(request, timeout=35) as response:
                payload = response.read()
        except HTTPError as error:
            parser.exit(1, f"{code}=F: HTTP {error.code}. Retry later or use an archived snapshot.\n")
        result, = json.loads(payload)["chart"]["result"]
        if result["meta"]["symbol"] != f"{code}=F" or not result.get("timestamp"):
            raise ValueError(f"Unexpected or empty chart for {code}=F")
        (args.out / f"{code}.json").write_bytes(payload)
        print(f"{code}=F: {len(result['timestamp'])} timestamps", flush=True)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
