# KPI Definitions

## Revenue Metrics
- **Total Revenue**: Sum of total_amount for all non-cancelled orders.
- **Monthly Recurring Revenue (MRR)**: Revenue from Subscriptions category in a given month.
- **Average Order Value (AOV)**: Total revenue / number of orders.
- **Revenue per Customer**: Total revenue / number of unique customers with orders.

## Profitability Metrics
- **Gross Profit**: Total revenue minus total cost of goods sold.
- **Gross Margin %**: (Gross profit / Total revenue) × 100.
- **Category Margin**: Gross margin calculated per product category.

## Customer Metrics
- **Customer Acquisition**: New customers (by signup_date) in a period.
- **Active Customer Rate**: Customers with is_active=True / total customers.
- **Customer Concentration**: Revenue share of top 10% of customers.
- **Segment Distribution**: Percentage of customers in each segment.

## Operational Metrics
- **Order Fulfillment Rate**: Delivered orders / total non-cancelled orders.
- **Cancellation Rate**: Cancelled orders / total orders.
- **Payment Success Rate**: Completed payments / total payments.
- **Inventory Turnover**: Units sold / average stock quantity.
- **Stockout Rate**: Products where stock_quantity = 0 / total products.

## Growth Metrics
- **Month-over-Month Revenue Growth**: (Current month revenue - prior month) / prior month × 100.
- **Quarter-over-Quarter Growth**: Same formula applied quarterly.
- **Category Growth**: Revenue growth calculated per product category.
