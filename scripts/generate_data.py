"""Generate synthetic multi-CSV business dataset for benchmarking.

Creates 6 interconnected tables:
- Customers: 200 customers across segments and regions
- Products: 50 products across categories
- Orders: ~2000 orders over 12 months (2024)
- Inventory: current stock levels per product
- Payments: one payment per order with status tracking
- Revenue: monthly revenue summary per product category

All data is deterministically seeded for reproducibility.
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
random.seed(SEED)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "csv"

# --- Reference data ---

SEGMENTS = ["Enterprise", "Mid-Market", "SMB", "Startup"]
REGIONS = ["North America", "Europe", "Asia-Pacific", "Latin America"]
INDUSTRIES = ["Technology", "Healthcare", "Finance", "Retail", "Manufacturing", "Education"]

PRODUCT_NAMES = {
    "Software": ["DataSync Pro", "CloudGuard", "AnalyticsSuite", "DevOps Toolkit",
                  "SecureVault", "CodeFlow IDE", "APIGateway Plus", "ML Pipeline",
                  "DocuManager", "FormBuilder"],
    "Hardware": ["ServerRack X1", "NetSwitch 5G", "StorageBox 10T", "EdgeNode Mini",
                 "PowerUnit 3K", "SmartSensor Hub", "DisplayPanel 4K", "RouterMesh Pro",
                 "BackupDrive 5T", "CableKit Premium"],
    "Services": ["Setup & Config", "24/7 Support Plan", "Training Package", "Data Migration",
                 "Security Audit", "Performance Tuning", "Custom Integration",
                 "Architecture Review", "Compliance Check", "Onboarding Program"],
    "Subscriptions": ["Basic Plan", "Pro Plan", "Enterprise Plan", "Team Plan",
                      "Developer Plan", "Analytics Add-on", "Storage Add-on",
                      "Security Add-on", "API Add-on", "Support Add-on"],
    "Accessories": ["USB-C Hub", "Laptop Stand", "Wireless Mouse", "Mech Keyboard",
                    "Monitor Arm", "Webcam HD", "Headset Pro", "Docking Station",
                    "Cable Organizer", "Screen Protector"],
}

PAYMENT_METHODS = ["Credit Card", "Wire Transfer", "ACH", "PayPal", "Invoice"]

FIRST_NAMES = [
    "James", "Maria", "Chen", "Priya", "Lucas", "Fatima", "Alex", "Yuki",
    "Omar", "Sofia", "Raj", "Emma", "Diego", "Aisha", "Noah", "Lena",
    "Kai", "Zara", "Ethan", "Mia", "Arjun", "Chloe", "Mateo", "Hana",
    "Leo", "Nina", "Sami", "Isla", "Ravi", "Julia",
]
LAST_NAMES = [
    "Smith", "Patel", "Wang", "Garcia", "Mueller", "Kim", "Johnson", "Silva",
    "Tanaka", "Brown", "Kumar", "Anderson", "Lopez", "Chen", "Williams",
    "Martinez", "Lee", "Taylor", "Nguyen", "Wilson", "Singh", "Thomas",
    "Robinson", "Clark", "Sato", "White", "Adams", "Scott", "Rivera", "Hall",
]
COMPANY_SUFFIXES = ["Inc", "Corp", "Ltd", "Group", "Solutions", "Technologies", "Systems"]
COMPANY_WORDS = [
    "Apex", "Nova", "Vertex", "Summit", "Quantum", "Pinnacle", "Nexus", "Horizon",
    "Atlas", "Vanguard", "Meridian", "Zenith", "Catalyst", "Forge", "Beacon",
    "Mosaic", "Prism", "Orbit", "Helix", "Pulse",
]


def generate_customers(n: int = 200) -> list[dict]:
    customers = []
    for i in range(1, n + 1):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        company = f"{random.choice(COMPANY_WORDS)} {random.choice(COMPANY_SUFFIXES)}"
        signup = date(2023, 1, 1) + timedelta(days=random.randint(0, 540))
        customers.append({
            "customer_id": f"C{i:04d}",
            "name": f"{first} {last}",
            "email": f"{first.lower()}.{last.lower()}{i}@{company.split()[0].lower()}.com",
            "company": company,
            "segment": random.choice(SEGMENTS),
            "region": random.choice(REGIONS),
            "industry": random.choice(INDUSTRIES),
            "signup_date": signup.isoformat(),
            "is_active": random.choices([True, False], weights=[85, 15])[0],
        })
    return customers


def generate_products() -> list[dict]:
    products = []
    pid = 1
    for category, names in PRODUCT_NAMES.items():
        for name in names:
            base_price = round(random.uniform(29.99, 4999.99), 2)
            cost = round(base_price * random.uniform(0.3, 0.7), 2)
            products.append({
                "product_id": f"P{pid:03d}",
                "name": name,
                "category": category,
                "price": base_price,
                "cost": cost,
                "margin_pct": round((base_price - cost) / base_price * 100, 1),
                "is_active": random.choices([True, False], weights=[90, 10])[0],
            })
            pid += 1
    return products


def generate_orders(customers: list[dict], products: list[dict], n: int = 2000) -> list[dict]:
    orders = []
    active_customers = [c for c in customers if c["is_active"]]
    active_products = [p for p in products if p["is_active"]]

    start_date = date(2024, 1, 1)
    end_date = date(2024, 12, 31)
    date_range = (end_date - start_date).days

    for i in range(1, n + 1):
        customer = random.choice(active_customers)
        product = random.choice(active_products)
        qty = random.choices([1, 2, 3, 5, 10], weights=[40, 25, 15, 10, 10])[0]
        order_date = start_date + timedelta(days=random.randint(0, date_range))
        discount_pct = random.choices([0, 5, 10, 15, 20], weights=[50, 20, 15, 10, 5])[0]
        subtotal = round(product["price"] * qty, 2)
        discount_amt = round(subtotal * discount_pct / 100, 2)
        total = round(subtotal - discount_amt, 2)

        orders.append({
            "order_id": f"O{i:05d}",
            "customer_id": customer["customer_id"],
            "product_id": product["product_id"],
            "order_date": order_date.isoformat(),
            "quantity": qty,
            "unit_price": product["price"],
            "discount_pct": discount_pct,
            "total_amount": total,
            "status": random.choices(
                ["delivered", "shipped", "processing", "cancelled"],
                weights=[60, 20, 12, 8],
            )[0],
        })
    return orders


def generate_inventory(products: list[dict]) -> list[dict]:
    inventory = []
    for p in products:
        stock = random.randint(0, 500)
        reorder = random.randint(10, 50)
        inventory.append({
            "product_id": p["product_id"],
            "product_name": p["name"],
            "stock_quantity": stock,
            "reorder_level": reorder,
            "reorder_needed": stock < reorder,
            "warehouse": random.choice(["Warehouse-A", "Warehouse-B", "Warehouse-C"]),
            "last_restocked": (date(2024, 12, 31) - timedelta(days=random.randint(1, 90))).isoformat(),
        })
    return inventory


def generate_payments(orders: list[dict]) -> list[dict]:
    payments = []
    for i, order in enumerate(orders, 1):
        order_date = date.fromisoformat(order["order_date"])
        if order["status"] == "cancelled":
            status = random.choice(["refunded", "failed"])
        else:
            status = random.choices(
                ["completed", "pending", "failed"],
                weights=[80, 15, 5],
            )[0]

        payment_date = order_date + timedelta(days=random.randint(0, 3))
        payments.append({
            "payment_id": f"PAY{i:05d}",
            "order_id": order["order_id"],
            "amount": order["total_amount"],
            "payment_method": random.choice(PAYMENT_METHODS),
            "payment_date": payment_date.isoformat(),
            "status": status,
        })
    return payments


def generate_revenue(orders: list[dict], products: list[dict]) -> list[dict]:
    """Monthly revenue summary per product category."""
    product_map = {p["product_id"]: p for p in products}
    buckets: dict[tuple[str, str], dict] = {}

    for order in orders:
        if order["status"] == "cancelled":
            continue
        product = product_map[order["product_id"]]
        od = date.fromisoformat(order["order_date"])
        month_key = od.strftime("%Y-%m")
        category = product["category"]
        key = (month_key, category)

        if key not in buckets:
            buckets[key] = {"revenue": 0.0, "cost": 0.0, "orders": 0, "units": 0}

        buckets[key]["revenue"] += order["total_amount"]
        buckets[key]["cost"] += product["cost"] * order["quantity"]
        buckets[key]["orders"] += 1
        buckets[key]["units"] += order["quantity"]

    revenue = []
    for (month, category), data in sorted(buckets.items()):
        revenue.append({
            "month": month,
            "category": category,
            "total_revenue": round(data["revenue"], 2),
            "total_cost": round(data["cost"], 2),
            "gross_profit": round(data["revenue"] - data["cost"], 2),
            "order_count": data["orders"],
            "units_sold": data["units"],
        })
    return revenue


def write_csv(filename: str, rows: list[dict]) -> None:
    path = OUTPUT_DIR / filename
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {filename}: {len(rows)} rows")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating business dataset...")
    customers = generate_customers(200)
    products = generate_products()
    orders = generate_orders(customers, products, 2000)
    inventory = generate_inventory(products)
    payments = generate_payments(orders)
    revenue = generate_revenue(orders, products)

    write_csv("customers.csv", customers)
    write_csv("products.csv", products)
    write_csv("orders.csv", orders)
    write_csv("inventory.csv", inventory)
    write_csv("payments.csv", payments)
    write_csv("revenue.csv", revenue)

    print("Done!")


if __name__ == "__main__":
    main()
