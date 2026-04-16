# Data Dictionary

## customers
| Column | Type | Description |
|--------|------|-------------|
| customer_id | TEXT PK | Unique customer identifier (C0001–C0200) |
| name | TEXT | Full name |
| email | TEXT | Email address |
| company | TEXT | Company name |
| segment | TEXT | Enterprise, Mid-Market, SMB, or Startup |
| region | TEXT | North America, Europe, Asia-Pacific, or Latin America |
| industry | TEXT | Technology, Healthcare, Finance, Retail, Manufacturing, or Education |
| signup_date | DATE | Date the customer registered |
| is_active | BOOLEAN | Whether the customer account is active |

## products
| Column | Type | Description |
|--------|------|-------------|
| product_id | TEXT PK | Unique product identifier (P001–P050) |
| name | TEXT | Product name |
| category | TEXT | Software, Hardware, Services, Subscriptions, or Accessories |
| price | REAL | Selling price per unit |
| cost | REAL | Cost of goods per unit |
| margin_pct | REAL | Profit margin percentage |
| is_active | BOOLEAN | Whether the product is currently offered |

## orders
| Column | Type | Description |
|--------|------|-------------|
| order_id | TEXT PK | Unique order identifier (O00001–O02000) |
| customer_id | TEXT FK | References customers.customer_id |
| product_id | TEXT FK | References products.product_id |
| order_date | DATE | Date the order was placed (2024-01-01 to 2024-12-31) |
| quantity | INTEGER | Number of units ordered |
| unit_price | REAL | Price per unit at time of order |
| discount_pct | REAL | Discount percentage applied (0, 5, 10, 15, or 20) |
| total_amount | REAL | Final amount after discount |
| status | TEXT | delivered, shipped, processing, or cancelled |

## inventory
| Column | Type | Description |
|--------|------|-------------|
| product_id | TEXT PK FK | References products.product_id |
| product_name | TEXT | Product name (denormalized) |
| stock_quantity | INTEGER | Current units in stock |
| reorder_level | INTEGER | Threshold that triggers reorder |
| reorder_needed | BOOLEAN | True if stock_quantity < reorder_level |
| warehouse | TEXT | Warehouse-A, Warehouse-B, or Warehouse-C |
| last_restocked | DATE | Date of last restock |

## payments
| Column | Type | Description |
|--------|------|-------------|
| payment_id | TEXT PK | Unique payment identifier |
| order_id | TEXT FK | References orders.order_id |
| amount | REAL | Payment amount |
| payment_method | TEXT | Credit Card, Wire Transfer, ACH, PayPal, or Invoice |
| payment_date | DATE | Date payment was processed |
| status | TEXT | completed, pending, failed, or refunded |

## revenue
| Column | Type | Description |
|--------|------|-------------|
| month | TEXT | Year-month (e.g., 2024-01) |
| category | TEXT | Product category |
| total_revenue | REAL | Sum of order totals for the month/category |
| total_cost | REAL | Sum of COGS for the month/category |
| gross_profit | REAL | total_revenue - total_cost |
| order_count | INTEGER | Number of orders |
| units_sold | INTEGER | Total units sold |
