from dagster import Definitions, load_assets_from_modules

from miracl import assets, enron, toutiao_news, hotpotqa  # noqa: TID252

all_assets = load_assets_from_modules([assets, enron, toutiao_news, hotpotqa])

defs = Definitions(
    assets=all_assets,
)
