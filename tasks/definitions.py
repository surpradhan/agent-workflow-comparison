"""Benchmark task definitions across all 4 difficulty levels.

Each task has a ground truth or evaluation criteria so workflows can be
scored objectively.
"""

from tasks.task_registry import Task, TaskLevel, registry

# =============================================================================
# Level 1: Retrieval — Simple lookups and searches
# =============================================================================

registry.register(Task(
    id="L1_T01",
    description="What is the total revenue for the Hardware category in March 2024?",
    level=TaskLevel.RETRIEVAL,
    expected_tools=["sql_query"],
    ground_truth="Look up revenue table: month='2024-03', category='Hardware', return total_revenue.",
    evaluation_criteria="Must return the exact total_revenue value from the revenue table.",
))

registry.register(Task(
    id="L1_T02",
    description="List all products in the Subscriptions category with their prices.",
    level=TaskLevel.RETRIEVAL,
    expected_tools=["sql_query"],
    ground_truth="SELECT name, price FROM products WHERE category='Subscriptions'",
    evaluation_criteria="Must return all 10 Subscriptions products with correct prices.",
))

registry.register(Task(
    id="L1_T03",
    description="How many customers are in the Enterprise segment?",
    level=TaskLevel.RETRIEVAL,
    expected_tools=["sql_query"],
    ground_truth="SELECT COUNT(*) FROM customers WHERE segment='Enterprise'",
    evaluation_criteria="Must return exact count of Enterprise customers.",
))

registry.register(Task(
    id="L1_T04",
    description="Which products need to be reordered based on current inventory levels?",
    level=TaskLevel.RETRIEVAL,
    expected_tools=["sql_query"],
    ground_truth="SELECT product_name FROM inventory WHERE reorder_needed=1",
    evaluation_criteria="Must list all products where stock is below reorder level.",
))

registry.register(Task(
    id="L1_T05",
    description="What is the discount policy for Enterprise customers?",
    level=TaskLevel.RETRIEVAL,
    expected_tools=["vector_search"],
    ground_truth="Enterprise segment customers are eligible for up to 20% discount.",
    evaluation_criteria="Must reference the 20% discount cap for Enterprise from business rules.",
))

registry.register(Task(
    id="L1_T06",
    description="What is the most expensive product in the Peripherals category?",
    level=TaskLevel.RETRIEVAL,
    expected_tools=["sql_query"],
    ground_truth="SELECT name, price FROM products WHERE category='Peripherals' ORDER BY price DESC LIMIT 1",
    evaluation_criteria="Must return the single highest-priced Peripherals product with its price.",
))

registry.register(Task(
    id="L1_T07",
    description="How many distinct customers placed at least one order in 2024?",
    level=TaskLevel.RETRIEVAL,
    expected_tools=["sql_query"],
    ground_truth="SELECT COUNT(DISTINCT customer_id) FROM orders WHERE strftime('%Y', order_date)='2024'",
    evaluation_criteria="Must return an exact count of unique customers with orders in 2024.",
))

registry.register(Task(
    id="L1_T08",
    description="What is the refund and return policy timeframe for physical hardware products?",
    level=TaskLevel.RETRIEVAL,
    expected_tools=["vector_search"],
    ground_truth="Hardware products have a 30-day return window from delivery date.",
    evaluation_criteria="Must cite the specific return window for hardware from the business rules.",
))

# =============================================================================
# Level 2: Analytical — Trends, aggregations, segmentation
# =============================================================================

registry.register(Task(
    id="L2_T01",
    description="What is the month-over-month revenue trend for 2024? Which month had the highest revenue?",
    level=TaskLevel.ANALYTICAL,
    expected_tools=["sql_query", "calculator"],
    ground_truth="Aggregate revenue by month, compute MoM change, identify peak month.",
    evaluation_criteria="Must show monthly totals, growth rates, and correctly identify the peak month.",
))

registry.register(Task(
    id="L2_T02",
    description="What is the average order value by customer segment? Which segment is most valuable?",
    level=TaskLevel.ANALYTICAL,
    expected_tools=["sql_query"],
    ground_truth="JOIN orders with customers, GROUP BY segment, compute AVG(total_amount).",
    evaluation_criteria="Must compute correct AOV per segment and identify the highest.",
))

registry.register(Task(
    id="L2_T03",
    description="What is the payment success rate by payment method?",
    level=TaskLevel.ANALYTICAL,
    expected_tools=["sql_query"],
    ground_truth="GROUP BY payment_method, calculate completed/total ratio per method.",
    evaluation_criteria="Must show success rate for each of the 5 payment methods.",
))

registry.register(Task(
    id="L2_T04",
    description="Which product categories have the highest and lowest gross margins?",
    level=TaskLevel.ANALYTICAL,
    expected_tools=["sql_query", "python_analysis"],
    ground_truth="Use revenue table to compute margin % by category, rank them.",
    evaluation_criteria="Must correctly rank all 5 categories by gross margin percentage.",
))

registry.register(Task(
    id="L2_T05",
    description="What is the order cancellation rate by region? Are there regional patterns?",
    level=TaskLevel.ANALYTICAL,
    expected_tools=["sql_query"],
    ground_truth="JOIN orders with customers, compute cancelled/total ratio per region.",
    evaluation_criteria="Must show cancellation rate per region and note any significant differences.",
))

registry.register(Task(
    id="L2_T06",
    description=(
        "What percentage of total 2024 revenue does each product category contribute? "
        "Rank from highest to lowest."
    ),
    level=TaskLevel.ANALYTICAL,
    expected_tools=["sql_query", "calculator"],
    ground_truth="SUM revenue by category, divide by grand total, rank descending.",
    evaluation_criteria="Must show correct percentage share for all 5 categories summing to 100%.",
))

registry.register(Task(
    id="L2_T07",
    description="Which customer segment places the highest volume of orders per customer?",
    level=TaskLevel.ANALYTICAL,
    expected_tools=["sql_query"],
    ground_truth="COUNT orders GROUP BY segment, JOIN customers, divide by customer count per segment.",
    evaluation_criteria="Must compute orders-per-customer for each segment and identify the highest.",
))

# =============================================================================
# Level 3: Multi-step reasoning — Explanations requiring multiple data sources
# =============================================================================

registry.register(Task(
    id="L3_T01",
    description=(
        "Revenue for Software dropped in Q3 2024 compared to Q2. "
        "Investigate and explain the possible reasons."
    ),
    level=TaskLevel.REASONING,
    expected_tools=["sql_query", "python_analysis", "vector_search"],
    ground_truth="Analyze order volume, pricing, cancellation rates, and inventory for Software in Q2 vs Q3.",
    evaluation_criteria=(
        "Must query multiple tables, compare Q2 vs Q3 metrics, and provide a reasoned "
        "explanation citing specific data points."
    ),
))

registry.register(Task(
    id="L3_T02",
    description=(
        "Identify the top 10 customers by total spending. What do they have in common? "
        "Are there patterns in their segment, region, or product preferences?"
    ),
    level=TaskLevel.REASONING,
    expected_tools=["sql_query", "python_analysis"],
    ground_truth="Aggregate orders by customer, join with customer details, analyze patterns.",
    evaluation_criteria=(
        "Must identify exact top 10, analyze their attributes, and describe "
        "meaningful patterns with supporting data."
    ),
))

registry.register(Task(
    id="L3_T03",
    description=(
        "The payment failure rate seems higher for certain order sizes. "
        "Analyze the relationship between order amount and payment success."
    ),
    level=TaskLevel.REASONING,
    expected_tools=["sql_query", "python_analysis", "calculator"],
    ground_truth="Join orders with payments, bucket by amount ranges, compute failure rates per bucket.",
    evaluation_criteria=(
        "Must segment orders by amount, compute payment failure rates per segment, "
        "and identify if there's a correlation."
    ),
))

registry.register(Task(
    id="L3_T04",
    description=(
        "Which warehouse is most at risk of stockouts in the next 30 days? "
        "Factor in current stock levels, reorder status, and recent sales velocity."
    ),
    level=TaskLevel.REASONING,
    expected_tools=["sql_query", "python_analysis", "vector_search"],
    ground_truth="Combine inventory data with recent order rates per warehouse, project stockout timeline.",
    evaluation_criteria=(
        "Must calculate sales velocity, project days-of-stock remaining per warehouse, "
        "and identify the highest-risk warehouse with reasoning."
    ),
))

registry.register(Task(
    id="L3_T05",
    description=(
        "Enterprise customers have a higher average order value than SMB customers, "
        "yet their payment failure rate is also higher. "
        "Analyze the data and explain this apparent paradox."
    ),
    level=TaskLevel.REASONING,
    expected_tools=["sql_query", "python_analysis"],
    ground_truth=(
        "Compare Enterprise vs SMB: order amounts, payment methods used, failure rates by method. "
        "Higher-value orders may correlate with riskier payment methods or stricter bank checks."
    ),
    evaluation_criteria=(
        "Must query both order values and payment failures by segment, "
        "identify the underlying driver (e.g. payment method mix or order size thresholds), "
        "and provide a data-supported explanation."
    ),
))

# =============================================================================
# Level 4: Decision-making — Strategy recommendations with trade-offs
# =============================================================================

registry.register(Task(
    id="L4_T01",
    description=(
        "Based on 2024 performance data, recommend which product category the company "
        "should invest in expanding next year. Justify with revenue, margin, and growth data."
    ),
    level=TaskLevel.DECISION,
    expected_tools=["sql_query", "python_analysis", "vector_search", "calculator"],
    ground_truth="Analyze all categories across revenue, margins, growth trends, and market position.",
    evaluation_criteria=(
        "Must analyze multiple dimensions (revenue, margin, growth, volume), "
        "acknowledge trade-offs, and make a defensible recommendation."
    ),
))

registry.register(Task(
    id="L4_T02",
    description=(
        "Design a discount optimization strategy. Current discounts seem to vary without "
        "clear ROI. Analyze discount effectiveness and propose a data-driven policy."
    ),
    level=TaskLevel.DECISION,
    expected_tools=["sql_query", "python_analysis", "vector_search", "calculator"],
    ground_truth="Analyze discount impact on order volume, revenue, and margins by segment.",
    evaluation_criteria=(
        "Must reference business rules, analyze discount-to-revenue relationship, "
        "and propose specific, actionable discount tiers with expected impact."
    ),
))

registry.register(Task(
    id="L4_T03",
    description=(
        "The company wants to reduce payment failures. Analyze the current failure patterns "
        "and recommend specific interventions with expected impact."
    ),
    level=TaskLevel.DECISION,
    expected_tools=["sql_query", "python_analysis", "vector_search"],
    ground_truth="Analyze failures by method, amount, region, segment; propose targeted fixes.",
    evaluation_criteria=(
        "Must identify failure patterns across dimensions, propose 3+ interventions, "
        "and estimate potential improvement for each."
    ),
))

registry.register(Task(
    id="L4_T04",
    description=(
        "Recommend a regional expansion strategy. Which region should the company "
        "prioritize for growth, and what product mix should they lead with?"
    ),
    level=TaskLevel.DECISION,
    expected_tools=["sql_query", "python_analysis", "vector_search", "calculator"],
    ground_truth="Compare regions on revenue, growth, customer base, product preferences.",
    evaluation_criteria=(
        "Must analyze all 4 regions comprehensively, consider multiple factors, "
        "and provide a specific recommendation with product mix rationale."
    ),
))

registry.register(Task(
    id="L4_T05",
    description=(
        "Should the company consolidate to fewer payment methods to reduce payment failures "
        "and operational complexity? Analyze the trade-offs and make a recommendation."
    ),
    level=TaskLevel.DECISION,
    expected_tools=["sql_query", "python_analysis", "vector_search"],
    ground_truth=(
        "Analyze each payment method by: transaction volume, success rate, revenue at risk, "
        "and customer segment preference. Weigh consolidation benefits vs. customer churn risk."
    ),
    evaluation_criteria=(
        "Must quantify revenue and volume per payment method, assess failure rates, "
        "reference business rules on payment policy, acknowledge trade-offs, "
        "and give a specific actionable recommendation with expected impact."
    ),
))
