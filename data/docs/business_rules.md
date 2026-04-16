# Business Rules

## Discount Policy
- Discounts are applied as a percentage of the subtotal (unit_price × quantity).
- Standard discount tiers: 0%, 5%, 10%, 15%, 20%.
- Discounts above 15% require manager approval.
- Enterprise segment customers are eligible for up to 20% discount.
- SMB and Startup segments are capped at 10% discount.

## Order Fulfillment
- Orders go through statuses: processing → shipped → delivered.
- Cancelled orders are excluded from revenue calculations.
- Orders not shipped within 5 business days are flagged as delayed.
- Refunds are processed within 7 business days of cancellation.

## Inventory Management
- Reorder is triggered when stock_quantity falls below reorder_level.
- Safety stock is maintained at 20% above the reorder level.
- Products in Warehouse-A serve North America and Latin America.
- Products in Warehouse-B serve Europe.
- Products in Warehouse-C serve Asia-Pacific.

## Customer Segmentation
- Enterprise: Annual revenue > $500,000 or company size > 1000 employees.
- Mid-Market: Annual revenue $50,000–$500,000.
- SMB: Annual revenue $10,000–$50,000.
- Startup: Annual revenue < $10,000 or company age < 2 years.

## Revenue Recognition
- Revenue is recognized on the delivery date, not the order date.
- Subscription revenue is recognized monthly over the subscription period.
- Services revenue is recognized upon completion of the service.
- Gross profit = revenue - cost of goods sold (COGS).
- Margin percentage = (revenue - COGS) / revenue × 100.
