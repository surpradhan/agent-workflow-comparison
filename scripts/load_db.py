"""Load CSV data into SQLite database with proper schema and indexes."""

import csv
import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_DIR = DATA_DIR / "csv"
DB_PATH = DATA_DIR / "db" / "benchmark.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    company TEXT NOT NULL,
    segment TEXT NOT NULL,
    region TEXT NOT NULL,
    industry TEXT NOT NULL,
    signup_date DATE NOT NULL,
    is_active BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    cost REAL NOT NULL,
    margin_pct REAL NOT NULL,
    is_active BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    product_id TEXT NOT NULL REFERENCES products(product_id),
    order_date DATE NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    discount_pct REAL NOT NULL,
    total_amount REAL NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inventory (
    product_id TEXT PRIMARY KEY REFERENCES products(product_id),
    product_name TEXT NOT NULL,
    stock_quantity INTEGER NOT NULL,
    reorder_level INTEGER NOT NULL,
    reorder_needed BOOLEAN NOT NULL,
    warehouse TEXT NOT NULL,
    last_restocked DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS payments (
    payment_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(order_id),
    amount REAL NOT NULL,
    payment_method TEXT NOT NULL,
    payment_date DATE NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS revenue (
    month TEXT NOT NULL,
    category TEXT NOT NULL,
    total_revenue REAL NOT NULL,
    total_cost REAL NOT NULL,
    gross_profit REAL NOT NULL,
    order_count INTEGER NOT NULL,
    units_sold INTEGER NOT NULL,
    PRIMARY KEY (month, category)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_product ON orders(product_id);
CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_payments_order ON payments(order_id);
CREATE INDEX IF NOT EXISTS idx_payments_status ON payments(status);
CREATE INDEX IF NOT EXISTS idx_customers_segment ON customers(segment);
CREATE INDEX IF NOT EXISTS idx_customers_region ON customers(region);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_revenue_month ON revenue(month);
"""


def str_to_bool(val: str) -> bool:
    return val.strip().lower() in ("true", "1", "yes")


def load_table(conn: sqlite3.Connection, table: str, filename: str) -> int:
    path = CSV_DIR / filename
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return 0

    columns = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in columns)
    col_names = ", ".join(columns)
    sql = f"INSERT OR REPLACE INTO {table} ({col_names}) VALUES ({placeholders})"

    # Type coercion for booleans
    bool_cols = {"is_active", "reorder_needed"}
    for row in rows:
        for col in bool_cols & row.keys():
            row[col] = str_to_bool(row[col])

    conn.executemany(sql, [tuple(row[c] for c in columns) for row in rows])
    return len(rows)


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing DB for clean load
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    print("Creating schema...")
    conn.executescript(SCHEMA)

    tables = [
        ("customers", "customers.csv"),
        ("products", "products.csv"),
        ("orders", "orders.csv"),
        ("inventory", "inventory.csv"),
        ("payments", "payments.csv"),
        ("revenue", "revenue.csv"),
    ]

    print("Loading data...")
    for table, filename in tables:
        count = load_table(conn, table, filename)
        print(f"  {table}: {count} rows")

    conn.commit()

    # Quick validation
    print("\nValidation:")
    for table, _ in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows in DB")

    conn.close()
    print(f"\nDatabase saved to {DB_PATH}")


if __name__ == "__main__":
    main()
