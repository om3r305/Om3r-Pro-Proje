# Polymarket domain classification, copied verbatim (patterns and first-match order) from
# namanhzz/prediction-market-calibration @ 143ca87, src/classify.py, so that "Politics" means
# exactly what it means in the paper under test (arXiv:2602.19520).
#
# MIT License — Copyright (c) 2025 the prediction-market-calibration authors.
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions: The above copyright
# notice and this permission notice shall be included in all copies or substantial portions of
# the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import re

POLYMARKET_DOMAIN_PATTERNS = [
    (re.compile(
        r"(?i)\b("
        r"NFL|NBA|MLB|NHL|UFC|MMA|boxing|tennis|golf|F1|Formula 1|NASCAR|"
        r"Super Bowl|World Series|Stanley Cup|NCAA|March Madness|"
        r"Premier League|Champions League|La Liga|Serie A|Bundesliga|Ligue 1|"
        r"Europa League|World Cup|Copa America|MLS|WNBA|PGA|ATP|WTA|"
        r"Grand Slam|Wimbledon|US Open|Australian Open|French Open|"
        r"Ryder Cup|Olympics|Olympic|medal|"
        r"playoff|postseason|All[- ]Star|MVP|Cy Young|Heisman|"
        r"touchdown|home run|strikeout|rushing yards|rebound|assist|"
        r"point spread|moneyline|sack|interception|three-pointer|"
        r"batting average|ERA|free throw|penalty kick|"
        r"49ers|Packers|Chiefs|Eagles|Cowboys|Patriots|Dolphins|Bills|Ravens|Bengals|"
        r"Lakers|Celtics|Warriors|Bucks|Nuggets|Knicks|76ers|Heat|Nets|Suns|"
        r"Yankees|Dodgers|Braves|Astros|Mets|Red Sox|Cubs|Phillies|Padres|"
        r"Maple Leafs|Bruins|Rangers|Lightning|Avalanche|Panthers|Oilers|"
        r"Arsenal|Liverpool|Manchester|Chelsea|Barcelona|Real Madrid|Bayern|Juventus|"
        r"game \d|win.*series|win.*championship|win.*title"
        r")\b"
    ), "Sports"),
    (re.compile(
        r"(?i)\b("
        r"president|presidential|election|senate|congress|governor|"
        r"democrat|republican|GOP|Trump|Biden|Obama|"
        r"vote|ballot|primary|caucus|nominee|nomination|"
        r"Supreme Court|executive order|impeach|filibuster|debt ceiling|"
        r"shutdown|electoral college|swing state|poll|approval rating|"
        r"cabinet|attorney general|secretary of state|Speaker|majority leader|"
        r"veto|legislation|midterm|runoff|recall|referendum|"
        r"RFK|DeSantis|Haley|Ramaswamy|Newsom|Pence|Kamala|Harris|"
        r"McConnell|Pelosi|AOC|Schumer|Vance|Vivek|"
        r"indictment|classified documents|"
        r"NATO|Ukraine|Russia|China.*Taiwan|Israel|Iran|sanction|"
        r"TikTok.*ban|government.*ban|Congress.*pass|"
        r"federal|POTUS|White House|inaugurat"
        r")\b"
    ), "Politics"),
    (re.compile(
        r"(?i)\b("
        r"Bitcoin|BTC|Ethereum|ETH|crypto|Solana|SOL|Dogecoin|DOGE|XRP|"
        r"token|blockchain|DeFi|altcoin|stablecoin|USDC|USDT|Tether|"
        r"Binance|Coinbase|FTX|SBF|halving|mining|NFT|airdrop|"
        r"memecoin|Pepe|Shiba|TVL|DEX|CEX|"
        r"Bitcoin ETF|Ethereum ETF|spot ETF|"
        r"Layer 2|rollup|zkSync|Arbitrum|Polygon|Avalanche|Cardano|ADA|"
        r"Ripple|Litecoin|Polkadot|Chainlink|Uniswap"
        r")\b"
    ), "Crypto"),
    (re.compile(
        r"(?i)\b("
        r"S&P|S&P 500|NASDAQ|Dow Jones|Russell 2000|stock|"
        r"Fed |Federal Reserve|interest rate|rate cut|rate hike|FOMC|"
        r"CPI|inflation|GDP|recession|tariff|"
        r"IPO|treasury|yield curve|bond|"
        r"unemployment|jobs report|nonfarm payroll|non-farm payroll|PCE|PPI|"
        r"housing starts|retail sales|consumer confidence|"
        r"earnings|revenue|market cap|PE ratio|"
        r"merger|acquisition|bankruptcy|credit rating|"
        r"oil price|gold price|commodity|forex|"
        r"quantitative|balance sheet|FDIC|"
        r"trade deficit|budget|fiscal|monetary policy"
        r")\b"
    ), "Finance"),
]


def classify_polymarket_domain(title: str) -> str:
    """Classify a Polymarket market title into a domain using regex patterns."""
    if not title:
        return "Other"
    for pattern, domain in POLYMARKET_DOMAIN_PATTERNS:
        if pattern.search(title):
            return domain
    return "Other"
