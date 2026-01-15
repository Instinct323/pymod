import re

import pandas as pd
import tbparse


def reduce(df: pd.DataFrame,
           agg_func: str = "last") -> pd.Series:
    return getattr(df.groupby("tag")["value"], agg_func)()


def filter_by_tag(ser: pd.Series,
                  re_pat: str) -> pd.Series:
    mask = [bool(re.search(re_pat, tag)) for tag in ser.index]
    return ser[mask]


if __name__ == '__main__':
    reader = tbparse.SummaryReader("/Lab/AnaGrasp/runs/v1/o0")
    scalars = reader.scalars
    print(filter_by_tag(reduce(scalars), r"^(test/.*/mAP_gold)|(epoch)"))
