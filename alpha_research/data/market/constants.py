"""Market-data constants used by the data-ingestion modules."""

# Mapping from the public string representation of a timeframe to its MT5
str_tf_to_mt5_tf: dict[str, int] = {
    'M1': 1,
    'M2': 2,
    'M3': 3,
    'M4': 4,
    'M5': 5,
    'M6': 6,
    'M10': 10,
    'M12': 12,
    'M15': 15,
    'M20': 20,
    'M30': 30,
    'H1': 16385,
    'H2': 16386,
    'H3': 16387,
    'H4': 16388,
    'H6': 16390,
    'H8': 16392,
    'H12': 16396,
    'D1': 16408,
    'W1': 32769,
    'MN1': 49153,
}
